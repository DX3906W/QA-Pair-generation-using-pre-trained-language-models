import os

import torch
import numpy as np
import torch.nn.functional as F

from .data_processor import MultitaskDataset
from .model import MultitaskModel
from transformers import AdamW
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from data_loader import *
from utils import evaluate_metrics


class MultitaskTrainer:
    def __init__(self,
                 lm,
                 generative_lm,
                 lm_name,
                 tokenizer,
                 lambda_p,
                 batch_size,
                 epochs,
                 lr,
                 vocab_size,
                 embed_dim,
                 num_heads,
                 dataset,
                 max_encoder_len=128,
                 max_decoder_len=256,
                 saved_model=None,
                 generation_task='answer',
                 ):
        self.lm = generative_lm.from_pretrained(lm_name)
        self.tokenizer = tokenizer.from_pretrained(lm_name)
        if 't5' in lm_name:
            self.tokenizer.add_special_tokens({'mask_token': "<mask>"})
        print('special tokens: ', self.tokenizer.all_special_tokens)
        print('vocab size: ', self.tokenizer.vocab_size)
        lm_vocab_path = './{lm_name}_vocab'.format(lm_name=lm_name)
        if not os.path.exists(lm_vocab_path):
            os.makedirs(lm_vocab_path)
        self.tokenizer.save_pretrained(lm_vocab_path)
        self.lm_name = lm_name
        self.benchmark_data = BenchmarkLoader().load_data('python_programming.json')

        self.lambda_p = lambda_p
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.vocab_size = vocab_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset.lower()
        self.saved_model = saved_model

        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

        self.model = MultitaskModel(self.lm, embed_dim, num_heads, vocab_size)
        self.model.to(self.device)
        self.optimizer = AdamW(params=self.model.parameters(), lr=self.lr)
        if self.saved_model is not None:
            self.load_model_from_ckpt()
        self.criterion = CrossEntropyLoss(ignore_index=-100)

        self.load_data()

    def load_model_from_ckpt(self):
        ckpt = torch.load(self.saved_model)
        self.model = ckpt['state_dict']
        self.optimizer.load_state_dict(ckpt['state_dict'])

    def load_data(self):
        if 'processed_squad' in self.dataset:
            train_data, val_data = SQuADLoader().get_data()
        elif 'race' in self.dataset:
            train_data, val_data = RACELoader().get_data()
        else:
            train_data, val_data = None, None
        train_data = MultitaskDataset(train_data, self.tokenizer,
                                      max_encoder_len=self.max_encoder_len,
                                      max_decoder_len=self.max_decoder_len)
        val_data = MultitaskDataset(val_data, self.tokenizer,
                                    max_encoder_len=self.max_encoder_len,
                                    max_decoder_len=self.max_decoder_len)

        self.train_dataloader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_data, batch_size=self.batch_size, shuffle=True)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            if epoch == 0:
                self.lambda_p = 1
            elif epoch == 1:
                self.lambda_p = 0
            elif epoch == 2:
                self.lambda_p = 0.5
            else:
                self.lambda_p = 0.2
            for step, data in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                batch = [d.to(self.device) for d in data]
                true_start_id, true_end_id = batch[3:]
                start_logits, end_logits, decoder_loss, _ = self.model(*batch[:3], mode='train')

                loss_start_idx = self.criterion(start_logits, true_start_id)
                loss_end_idx = self.criterion(end_logits, true_end_id)
                loss = self.lambda_p * decoder_loss + (1 - self.lambda_p) * (loss_start_idx + loss_end_idx)
                loss.backward()
                self.optimizer.step()

                if step % 10 == 0:
                    print("Epoch: {}  Step:{}  Loss:{}".format(epoch, step, loss.item()))
            path = './saved_models/multitask/{lm_name}'.format(lm_name=self.lm_name)
            folder = os.path.exists(path)
            if not folder:
                os.makedirs(path)

            torch.save({'state_dict': self.model, 'optimizer': self.optimizer},
                       '{path}/multi_{epoch}.pth.tar'.format(
                           path=path, lm_name=self.lm_name, epoch=epoch))
            # self.validate()
            # print(self.infer(epoch, 'greedy_decode'))

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for step, data in enumerate(self.val_dataloader):
                batch = [d.to(self.device) for d in data]
                true_start_id, true_end_id = batch[3:]
                start_logits, end_logits, decoder_loss, _ = self.model(*batch[:3], mode='valid')

                loss_start_idx = self.criterion(start_logits, true_start_id)
                loss_end_idx = self.criterion(end_logits, true_end_id)
                loss = self.lambda_p * decoder_loss + (1 - self.lambda_p) * (loss_start_idx + loss_end_idx)

                if step % 10 == 0:
                    print("Step:{}  Loss:{}".format(step, loss.item()))

    def infer(self, epoch, save_predictions=False):
        self.model.eval()
        predictions = []
        references = []
        for passage, answer, question in zip(self.benchmark_data['passage'], self.benchmark_data['answer'],
                                             self.benchmark_data['question']):
            references.append(question + ' ' + answer)
            p_inputs = self.tokenizer.encode_plus(passage,
                                                  return_tensors='pt',
                                                  padding='max_length',
                                                  truncation=True,
                                                  max_length=self.max_encoder_len)
            p_input_ids = p_inputs['input_ids'].to(self.device)
            p_attention_mask = p_inputs['attention_mask'].to(self.device)
            g_q_encode = self.model.generate_question(p_input_ids)
            g_q = self.tokenizer.decode(g_q_encode.squeeze().tolist(), skip_special_tokens=True)

            start_logits, end_logits, decoder_logits, _ = self.model(p_input_ids, p_attention_mask, None, None)
            start_idx = torch.argmax(start_logits, dim=1)[0].item()
            end_idx = torch.argmax(end_logits, dim=1)[0].item()
            g_a = passage[start_idx - 1:end_idx - 1]
            predictions.append(g_q + ' ' + g_a)

        if save_predictions:
            with open('../benchmark_qa/multi_task/{lm_name}/multi_{lm_name}_{epoch}.txt'.format(
                    lm_name=self.lm_name, epoch=epoch), 'a') as f:
                for pre in predictions:
                    f.write(pre)
            f.close()
        return evaluate_metrics(predictions, references)


class MultitaskGenerator:
    def __init__(self,
                 lm_name,
                 tokenizer,
                 max_encoder_len=128,
                 max_decoder_len=64,
                 saved_model=None
                 ):
        self.tokenizer = tokenizer.from_pretrained(lm_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.saved_model = saved_model

        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

        self.model = torch.load(saved_model)['state_dict']
        self.model.to(self.device)

    def generate(self, passage):
        self.model.eval()
        p_inputs = self.tokenizer.encode_plus(passage,
                                              return_tensors='pt',
                                              padding='max_length',
                                              truncation=True,
                                              max_length=self.max_encoder_len)
        p_input_ids = p_inputs['input_ids'].to(self.device)
        p_attention_mask = p_inputs['attention_mask'].to(self.device)
        g_q_encode = self.model.generate_question(p_input_ids)
        g_q = self.tokenizer.decode(g_q_encode.squeeze().tolist(), skip_special_tokens=True)

        start_logits, end_logits, decoder_logits, _ = self.model(p_input_ids, p_attention_mask, None, mode='valid')
        start_idx = torch.argmax(start_logits, dim=1)[0].item()
        end_idx = torch.argmax(end_logits, dim=1)[0].item()
        g_a_encode = p_input_ids[0][start_idx:end_idx]
        g_a = self.tokenizer.decode(g_a_encode)

        return g_q, g_a
