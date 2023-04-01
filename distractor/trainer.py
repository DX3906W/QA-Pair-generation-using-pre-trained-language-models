import os
import torch
from torch.utils.data import DataLoader
from transformers import AdamW

from data_loader import *
from .data_processor import DistractorDataset
from .model import DistractorGenerationModel
from utils import *


class DGTrainer:
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
                 saved_model=None
                 ):
        self.tokenizer = tokenizer.from_pretrained(lm_name)
        self.benchmark_data = BenchmarkLoader().load_data('python_programming.txt')
        self.max_encoder_len = max_encoder_len
        self.lm_name = lm_name
        self.saved_model = saved_model

        self.lambda_p = lambda_p
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.vocab_size = vocab_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset.lower()

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
        data_loader = DGRACELoader()
        train_data = data_loader.load_data('race_train_original.json')
        dev_data = data_loader.load_data('race_dev_original.json')
        test_data = data_loader.load_data('race_test_original.json')

        train_dataset = DistractorDataset(train_data, self.tokenizer)
        val_dataset = DistractorDataset(dev_data, self.tokenizer)
        test_dataset = DistractorDataset(test_data, self.tokenizer)

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for step, data in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                batch = [d.to(self.device) for d in data]
                p_input_ids, p_attention_mask, d_input_ids = batch
                outputs = self.model(p_input_ids, p_attention_mask, d_input_ids)

                loss = outputs[0]
                loss.backward()
                self.optimizer.step()

                if step % 10 == 0:
                    print("Epoch: {}  Step:{}  Loss:{}".format(epoch, step, loss.item()))
            path = './saved_models/distractor/{lm_name}'.format(lm_name=self.lm_name)
            folder = os.path.exists(path)
            if not folder:
                print('creat path')
                os.makedirs(path)
            torch.save({'state_dict': self.model, 'optimizer': self.optimizer},
                       './{path}/{epoch}.pth.tar'.format(path=path, epoch=epoch))
            self.validate()
            print(self.infer())

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for step, data in enumerate(self.val_dataloader):
                batch = [d.to(self.device) for d in data]
                p_input_ids, p_attention_mask, a_input_ids = batch
                outputs = self.model(p_input_ids, p_attention_mask, a_input_ids)
                loss = outputs[0]
                if step % 10 == 0:
                    print(" Step:{}  Loss:{}".format(step, loss.item()))

    def infer(self, save_predictions=False):
        self.model.eval()
        predictions = []
        references = []
        for passage, answer, question, distractor in zip(self.benchmark_data['passage'], self.benchmark_data['answer'],
                                                         self.benchmark_data['question'],
                                                         self.benchmark_data['distractor']):
            input_text = 'Generated distractor: ' + question + 'answer: ' + answer + 'context: ' + passage + ' </s>'
            references.append(distractor)
            encode_inputs = self.tokenizer.encode_plus(input_text,
                                                       return_tensors="pt",
                                                       padding='max_length',
                                                       truncation=True,
                                                       max_length=self.max_encoder_len)
            with torch.no_grad():
                input_ids, attention_mask = encode_inputs['input_ids'], encode_inputs['attention_mask']
                input_ids = input_ids.to(self.device)
                outputs = self.model.generate(input_ids)
                decoded_outputs = self.tokenizer.decode(outputs.squeeze().tolist(), skip_special_tokens=True)
                predictions.append(decoded_outputs)
        return evaluate_metrics(predictions, references)
