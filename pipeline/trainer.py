import os
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import AutoTokenizer

from data_loader import *
from .data_processor import AGDataset, QGDataset
from utils import evaluate_metrics


class AGQGTrainer:
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
                 max_decoder_len=64,
                 saved_model=None,
                 generation_task='answer',
                 ):
        self.tokenizer = tokenizer.from_pretrained(lm_name)
        self.generation_task = generation_task
        self.benchmark_data = BenchmarkLoader().load_data('python_programming.txt')
        print(self.benchmark_data)

        self.lm_name = lm_name

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

        self.model = generative_lm.from_pretrained(lm_name)
        self.optimizer = AdamW(params=self.model.parameters(), lr=self.lr)
        if self.saved_model is not None:
            self.load_model_from_ckpt()
        self.model.to(self.device)

        self.load_data()

    def load_model_from_ckpt(self):
        ckpt = torch.load(self.saved_model)
        self.model.load_state_dict(ckpt['state_dict'])
        self.optimizer.load_state_dict(ckpt['state_dict'])

    def load_data(self):
        if 'processed_squad' in self.dataset:
            train_data, val_data = SQuADLoader().get_data()
        elif 'race' in self.dataset:
            train_data, val_data = RACELoader().get_data()
        else:
            train_data, val_data = None, None
        if self.generation_task == 'answer':
            train_dataset = AGDataset(train_data, self.tokenizer)
            val_dataset = AGDataset(val_data, self.tokenizer)
        else:
            train_dataset = QGDataset(train_data, self.tokenizer)
            val_dataset = QGDataset(val_data, self.tokenizer)

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for step, data in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                batch = [d.to(self.device) for d in data]
                input_ids, attention_mask, label_ids = batch
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     labels=label_ids,)
                loss = outputs[0]
                loss.backward()
                self.optimizer.step()

                if step % 10 == 0:
                    print("Epoch: {}  Step:{}  Loss:{}".format(epoch, step, loss.item()))
            path = './saved_models/pipeline/{lm_name}'.format(generation_task=self.generation_task, lm_name=self.lm_name)
            folder = os.path.exists(path)
            if not folder:
                print('creat path')
                os.makedirs(path)
            torch.save({'state_dict': self.model, 'optimizer': self.optimizer},
                       '{path}/{generation_task}_{lm_name}_{epoch}.pth.tar'.format(
                           path=path, generation_task=self.generation_task, lm_name=self.lm_name, epoch=epoch))
            self.validate()
            self.infer()

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for step, data in enumerate(self.val_dataloader):
                batch = [d.to(self.device) for d in data]
                input_ids, attention_mask, label_ids = batch
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     labels=label_ids,)
                loss = outputs[0]
                if step % 10 == 0:
                    print("Validation Step:{}  Loss:{}".format(step, loss.item()))

    def infer(self, save_predictions=False):
        self.model.eval()
        predictions = []
        references = []
        for passage, answer, question in zip(self.benchmark_data['passage'], self.benchmark_data['answer'], self.benchmark_data['question']):
            if self.generation_task == 'answer':
                inputs = passage
                references.append(answer)
            else:
                inputs = passage + ' [SEP] ' + answer
                references.append(question)
            encode_inputs = self.tokenizer.encode_plus(inputs,
                                                       return_tensors="pt",
                                                       padding='max_length',
                                                       truncation=True,
                                                       max_length=self.max_encoder_len)
            with torch.no_grad():
                input_ids, attention_mask = encode_inputs['input_ids'], encode_inputs['attention_mask']
                outputs = self.model.generate(input_ids)
                decoded_outputs = self.tokenizer.decode(outputs.squeeze().tolist(), skip_special_tokens=True)
                predictions.append(decoded_outputs)
        return evaluate_metrics(predictions, references)


class PipelineGenerator:
    def __init__(self, lm, lm_name, tokenizer, saved_ag_model, saved_qg_model, max_encoder_len):
        self.ag_model = lm.from_pretrained(saved_ag_model)
        self.qg_model = lm.from_pretrained(saved_qg_model)
        self.tokenizer = tokenizer.from_pretrained(lm_name)
        self.max_encoder_len = max_encoder_len

    def generate(self, p):
        with torch.no_grad():
            p_encode = self.tokenizer.encode_plus(p,
                                                  return_tensors="pt",
                                                  padding="max_length",
                                                  truncation=True,
                                                  max_length=self.max_encoder_len)
            p_input_ids, p_attention_mask = p_encode['input_ids'][0], p_encode['attention_mask'][0]
            g_a_encode = self.ag_model.generate(p_input_ids, p_attention_mask)
            g_a = self.tokenizer.decode(g_a_encode.squeeze().tolist(), skip_special_tokens=True)

            pa = p + ' [SEP] ' + g_a
            pa_encode = self.tokenizer.encode_plus(pa,
                                                   return_tensors="pt",
                                                   padding="max_length",
                                                   truncation=True,
                                                   max_length=self.max_encoder_len)
            pa_input_ids, pa_attention_mask = pa_encode['input_ids'][0], pa_encode['attention_mask'][0]
            g_q_encode = self.qg_model.generate(pa_input_ids, pa_attention_mask)
            g_q = self.tokenizer.decode(g_q_encode.squeeze().tolist(), skip_special_tokens=True)

            return g_a, g_q
