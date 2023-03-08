import torch
from torch.utils.data import DataLoader
from transformers import AdamW

from data_loader import *
from data_processor import AGDataset, QGDataset
from model import AnswerGenerationModel, QuestionGenerationModel
from utils import *


class AGTrainer:
    def __init__(self,
                 lm,
                 tokenizer,
                 lambda_p,
                 batch_size,
                 epochs,
                 lr,
                 vocab_size,
                 dataset
                 ):
        self.lm = lm
        self.tokenizer = tokenizer

        self.lambda_p = lambda_p
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.vocab_size = vocab_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset.lower()

        self.model = AnswerGenerationModel(self.lm)
        self.optimizer = AdamW(params=self.model.parameters(), lr=self.lr)

        self.load_data()

    def load_data(self):
        if 'squad' in self.dataset:
            data = SQuADLoader().get_data()
        elif 'race' in self.dataset:
            data = RACELoader().get_data()
        else:
            data = None
        train_data, val_data, test_data = split_dataset(data)
        train_dataset = AGDataset(train_data, self.tokenizer)
        val_dataset = AGDataset(val_data, self.tokenizer)
        test_dataset = AGDataset(test_data, self.tokenizer)

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for step, data in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                batch = [d.to(self.device) for d in data]
                p_input_ids, p_attention_mask, a_input_ids = batch[4:]
                outputs = self.model(p_input_ids, p_attention_mask, a_input_ids)

                loss = outputs[0]
                loss.backward()
                self.optimizer.step()

                if step % 10 == 0:
                    print("Epoch: {}  Step:{}  Loss:{}".format(epoch, step, loss.item()))

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for step, data in enumerate(self.val_dataloader):
                batch = [d.to(self.device) for d in data]
                p_input_ids, p_attention_mask, a_input_ids = batch[4:]
                outputs = self.model(p_input_ids, p_attention_mask, a_input_ids)
                loss = outputs[0]
                if step % 10 == 0:
                    print(" Step:{}  Loss:{}".format(step, loss.item()))

    def infer(self, p):
        self.model.eval()
        decoded_outputs = []
        p_encode = self.tokenizer.encode_plus(p,
                                              return_tensors="pt",
                                              padding="max_length",
                                              truncation=True,
                                              max_length=self.max_encoder_len)
        with torch.no_grad():
            p_input_ids, p_attention_mask = p_encode['input_ids'][0], p_encode['attention_mask'][0]
            outputs = self.model.generate(p_input_ids, p_attention_mask)

            decoded = self.tokenizer.decode(outputs.squeeze().tolist(), skip_special_tokens=True)
            decoded_outputs.append(decoded)

        return decoded_outputs


class QGTrainer:
    def __init__(self,
                 lm,
                 tokenizer,
                 lambda_p,
                 batch_size,
                 epochs,
                 lr,
                 vocab_size,
                 dataset
                 ):
        self.lm = lm
        self.tokenizer = tokenizer

        self.lambda_p = lambda_p
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.vocab_size = vocab_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset.lower()

        self.model = QuestionGenerationModel(self.lm)
        self.optimizer = AdamW(params=self.model.parameters(), lr=self.lr)

        self.load_data()

    def load_data(self):
        if 'squad' in self.dataset:
            data = SQuADLoader().get_data()
        elif 'race' in self.dataset:
            data = RACELoader().get_data()
        else:
            data = None
        train_data, val_data, test_data = split_dataset(data)
        train_dataset = QGDataset(train_data, self.tokenizer)
        val_dataset = QGDataset(val_data, self.tokenizer)
        test_dataset = QGDataset(test_data, self.tokenizer)

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for step, data in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                batch = [d.to(self.device) for d in data]
                p_input_ids, p_attention_mask, a_input_ids = batch[4:]
                outputs = self.model(p_input_ids, p_attention_mask, a_input_ids)

                loss = outputs[0]
                loss.backward()
                self.optimizer.step()

                if step % 10 == 0:
                    print("Epoch: {}  Step:{}  Loss:{}".format(epoch, step, loss.item()))

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for step, data in enumerate(self.val_dataloader):
                batch = [d.to(self.device) for d in data]
                p_input_ids, p_attention_mask, a_input_ids = batch[4:]
                outputs = self.model(p_input_ids, p_attention_mask, a_input_ids)
                loss = outputs[0]
                if step % 10 == 0:
                    print(" Step:{}  Loss:{}".format(step, loss.item()))

    def infer(self, p, a):
        self.model.eval()
        pa_decode = p + ' [SEP] ' + a
        with torch.no_grad():
            p_input_ids, p_attention_mask = pa_decode['input_ids'][0], pa_decode['attention_mask'][0]
            outputs = self.model.generate(p_input_ids, p_attention_mask)

            decoded_outputs = self.tokenizer.decode(outputs.squeeze().tolist(), skip_special_tokens=True)

        return decoded_outputs
