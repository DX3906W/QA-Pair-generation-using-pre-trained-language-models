import os
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from .model import QuestionGenerationModel, KeyphraseGenerationModel, AnswerGenerationModel
from data_loader import SQuADLoaderForJoint, BenchmarkLoader, RACELoader
from .data_processor import AGDataset, QGKGDataset
from utils import evaluate_metrics


class QGKGTrainer:
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
        lm_vocab_path = './{lm_name}_vocab'.format(lm_name=lm_name)
        if not os.path.exists(lm_vocab_path):
            os.makedirs(lm_vocab_path)
        if 't5' in lm_name:
            self.tokenizer.add_special_tokens({'cls_token': "<cls>"})
        self.tokenizer.save_pretrained(lm_vocab_path)
        print('vocab size: ', self.tokenizer.vocab_size)
        print('special tokens: ', self.tokenizer.all_special_tokens)
        self.benchmark_data = BenchmarkLoader().load_data('python_programming.json')
        # print(self.benchmark_data)

        self.lm_name = lm_name

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset.lower()
        self.saved_model = saved_model

        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

        self.qg_model = QuestionGenerationModel(generative_lm, lm_name)
        self.kg_model = KeyphraseGenerationModel(generative_lm, lm_name)
        self.qg_optimizer = AdamW(params=self.qg_model.parameters(), lr=self.lr)
        self.kg_optimizer = AdamW(params=self.kg_model.parameters(), lr=self.lr)

        if self.saved_model is not None:
            self.load_model_from_ckpt()
        self.qg_model.to(self.device)
        self.kg_model.to(self.device)

        self.test_sample = 'A modern computer can be defined as a machine that stores and manipulates information under the control of a  changeable program.'
        self.load_data()

    def load_model_from_ckpt(self):
        ckpt = torch.load(self.saved_model)
        self.model = ckpt['state_dict']
        self.qg_optimizer = ckpt['optimizer']

    def load_data(self):
        if 'processed_squad' in self.dataset:
            train_data, val_data = SQuADLoaderForJoint().get_data()
        elif 'race' in self.dataset:
            train_data, val_data = RACELoader().get_data()
        else:
            train_data, val_data = None, None
        train_dataset = QGKGDataset(train_data, self.tokenizer)
        val_dataset = QGKGDataset(val_data, self.tokenizer)

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True)

    def _prepare_input_for_kg(self, passage, keyphrase):
        encoder_inputs = self.tokenizer(
            passage,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_encoder_len
        )
        encoder_input_ids = encoder_inputs["input_ids"].to(self.device)
        encoder_attention_mask = encoder_inputs["attention_mask"].to(self.device)
        
        if keyphrase is None:
            return encoder_input_ids, encoder_attention_mask, None, None

        decoder_inputs = self.tokenizer(
            keyphrase,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_encoder_len
        )
        decoder_input_ids = decoder_inputs["input_ids"].to(self.device)
        decoder_attention_mask = decoder_inputs["attention_mask"].to(self.device)
        
        return encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask

    def _prepare_input_for_qg(self, keyphrase, passage, question):
        input_text = []
        for i in range(self.batch_size):
            input_text.append(keyphrase[i] + " " + self.tokenizer.cls_token + " " + passage[i])
        encoder_inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_encoder_len
        )
        encoder_input_ids = encoder_inputs["input_ids"].to(self.device)
        encoder_attention_mask = encoder_inputs["attention_mask"].to(self.device)

        decoder_inputs = self.tokenizer(
            question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_encoder_len
        )
        decoder_input_ids = decoder_inputs["input_ids"].to(self.device)
        decoder_attention_mask = decoder_inputs["attention_mask"].to(self.device)

        return encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask

    def _decode_output(self, output_encode):
        print(output_encode.shape)
        return self.tokenizer.batch_decode(output_encode, skip_special_tokens=True)

    def train(self):
        self.kg_model.train()
        self.qg_model.train()
        for epoch in range(self.epochs):
            for step, data in enumerate(self.train_dataloader):
                passage, question, answer = data
                for iter in range(10):
                    print(iter)
                    if iter == 0:
                        with torch.no_grad():
                            encoder_input_ids, encoder_attention_mask, _, _ = self._prepare_input_for_kg(passage, None)
                            _, k_encode = self.kg_model(encoder_input_ids,
                                                        encoder_attention_mask,
                                                        decoder_input_ids=None,
                                                        question_hidden_state=None,
                                                        mode='infer')
                            # k_encode = k_encode.detach()
                            keyphrase = self._decode_output(k_encode)

                    self.qg_optimizer.zero_grad()
                    encoder_input_ids, encoder_attention_mask, decoder_input_ids, _ = self._prepare_input_for_qg(
                        keyphrase, passage, question)
                    decoder_last_hidden_state, qg_loss, decoder_out = self.qg_model(
                        encoder_input_ids,
                        encoder_attention_mask,
                        decoder_input_ids,
                        mode='train')
                    qg_loss.backward()
                    self.qg_optimizer.step()

                    self.kg_optimizer.zero_grad()
                    encoder_input_ids, encoder_attention_mask, decoder_input_ids, _ = self._prepare_input_for_kg(
                        keyphrase, passage)
                    kg_loss, decoder_out = self.kg_model(
                        encoder_input_ids,
                        encoder_attention_mask,
                        decoder_input_ids,
                        decoder_last_hidden_state.detach(),
                        mode='train')
                    keyphrase = self._decode_output(decoder_out)
                    kg_loss.backward()
                    self.kg_optimizer.step()

                    if step % 10 == 0 and iter == 9:
                        print("Epoch: {}  Step:{}  KG Loss: {}   QG Loss: {}".format(
                            epoch, step, kg_loss.item(), qg_loss.item()))
            path = './saved_models/joint/{lm_name}'.format(lm_name=self.lm_name)
            folder = os.path.exists(path)
            if not folder:
                print('creat path')
                os.makedirs(path)
            torch.save({'state_dict': self.kg_model, 'optimizer': self.kg_optimizer},
                       '{path}/kg_{epoch}.pth.tar'.format(
                           path=path, epoch=epoch))
            torch.save({'state_dict': self.qg_model, 'optimizer': self.qg_optimizer},
                       '{path}/qg_{epoch}.pth.tar'.format(
                           path=path, epoch=epoch))

            self.validate()
            # print(self.infer())

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for step, data in enumerate(self.val_dataloader):
                batch = [d.to(self.device) for d in data]
                input_ids, attention_mask, label_ids = batch
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     labels=label_ids, )
                loss = outputs[0]
                if step % 10 == 0:
                    print("Validation Step:{}  Loss:{}".format(step, loss.item()))

                if step % 50 == 0:
                    input_ids = self.tokenizer(self.test_sample).input_ids
                    g_p = self.model.generate(input_ids)
                    print(self.tokenizer.decode(g_p.squeeze().tolist(), skip_special_tokens=True))

    # def infer(self, save_predictions=False):
    #     self.model.eval()
    #     predictions = []
    #     references = []
    #     for passage, answer, question in zip(self.benchmark_data['passage'], self.benchmark_data['answer'],
    #                                          self.benchmark_data['question']):
    #         inputs = passage
    #         references.append(answer)
    #         with torch.no_grad():
    #             input_ids, attention_mask = encode_inputs['input_ids'], encode_inputs['attention_mask']
    #             input_ids = input_ids.to(self.device)
    #             outputs = self.model.generate(input_ids)
    #             decoded_outputs = self.tokenizer.decode(outputs.squeeze().tolist(), skip_special_tokens=True)
    #             predictions.append(decoded_outputs)
    #     return evaluate_metrics(predictions, references)


class AGTrainer:
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
        lm_vocab_path = './{lm_name}_vocab'.format(lm_name=lm_name)
        if not os.path.exists(lm_vocab_path):
            os.makedirs(lm_vocab_path)
        self.tokenizer.save_pretrained(lm_vocab_path)
        print('vocab size: ', self.tokenizer.vocab_size)
        print('special tokens: ', self.tokenizer.all_special_tokens)
        self.benchmark_data = BenchmarkLoader().load_data('python_programming.txt')
        # print(self.benchmark_data)

        self.lm_name = lm_name

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset.lower()
        self.saved_model = saved_model

        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

        self.ag_model = AnswerGenerationModel(generative_lm, lm_name)
        self.ag_optimizer = AdamW(params=self.ag_model.parameters(), lr=self.lr)
        self.qgkg_generator = QGKGGenerator(
            lm_name,
            tokenizer,
            'saved_qg_model',
            'saved_kg_model',
            max_encoder_len,
            max_decoder_len)

        if self.saved_model is not None:
            self.load_model_from_ckpt()
        self.model.to(self.device)

        self.test_sample = 'A modern computer can be defined as a machine that stores and manipulates information under the control of a  changeable program.'
        self.load_data()

    def load_model_from_ckpt(self):
        ckpt = torch.load(self.saved_model)
        self.model = ckpt['state_dict']
        self.ag_optimizer.load_state_dict(ckpt['state_dict'])

    def load_data(self):
        if 'processed_squad' in self.dataset:
            train_data, val_data = SQuADLoaderForJoint().get_data()
        elif 'race' in self.dataset:
            train_data, val_data = RACELoader().get_data()
        else:
            train_data, val_data = None, None
        train_dataset = QGKGDataset(train_data, self.tokenizer)
        val_dataset = QGKGDataset(val_data, self.tokenizer)

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True)

    def _prepare_input_for_ag(self, keyphrase, passage, question, answer):
        input_text = []
        for i in range(self.batch_size):
            input_text.append(keyphrase[i] + ' {} '.format(self.tokenizer.cls_token) + passage[i] + \
                              ' {} '.format(self.tokenizer.sep_token) + question[i])
        encoder_inputs = self.tokenizer.encode_plus(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_encoder_len
        )
        encoder_input_ids = encoder_inputs["input_ids"][0]
        encoder_attention_mask = encoder_inputs["attention_mask"][0]

        decoder_inputs = self.tokenizer.encode_plus(
            answer,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_encoder_len
        )
        decoder_input_ids = decoder_inputs["input_ids"][0]
        decoder_attention_mask = decoder_inputs["attention_mask"][0]

        encoder_input_ids = encoder_input_ids.to(self.device)
        encoder_attention_mask = encoder_attention_mask.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)
        decoder_attention_mask = decoder_attention_mask.to(self.device)

        return encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask

    def _decode_output(self, output_encode):
        return self.tokenizer.decode(output_encode.squeeze().tolist(), skip_special_tokens=True)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for step, data in enumerate(self.train_dataloader):
                passage, _, answer = data
                self.ag_optimizer.zero_grad()
                keyphrase, question = self.qgkg_generator.generate(passage)
                encoder_input_ids, encoder_attention_mask, decoder_input_ids, _ = self._prepare_input_for_ag(
                    keyphrase, passage, question, answer)
                decoder_last_hidden_state, ag_loss, decoder_out = self.ag_model(
                    encoder_input_ids,
                    encoder_attention_mask,
                    decoder_input_ids,
                    mode='train')
                ag_loss.backward()
                self.ag_optimizer.step()

                if step % 10 == 0 and iter == 9:
                    print("Epoch: {}  Step:{}  AG Loss: {}".format(
                        epoch, step, ag_loss.item()))
            path = './saved_models/joint/{lm_name}'.format(lm_name=self.lm_name)
            folder = os.path.exists(path)
            if not folder:
                print('creat path')
                os.makedirs(path)
            torch.save({'state_dict': self.ag_model, 'optimizer': self.ag_optimizer},
                       '{path}/ag_{epoch}.pth.tar'.format(
                           path=path, epoch=epoch))
            self.validate()
            # print(self.infer())

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for step, data in enumerate(self.val_dataloader):
                batch = [d.to(self.device) for d in data]
                input_ids, attention_mask, label_ids = batch
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     labels=label_ids, )
                loss = outputs[0]
                if step % 10 == 0:
                    print("Validation Step:{}  Loss:{}".format(step, loss.item()))

                if step % 50 == 0:
                    input_ids = self.tokenizer(self.test_sample).input_ids
                    g_p = self.model.generate(input_ids)
                    print(self.tokenizer.decode(g_p.squeeze().tolist(), skip_special_tokens=True))

    # def infer(self, save_predictions=False):
    #     self.model.eval()
    #     predictions = []
    #     references = []
    #     for passage, answer, question in zip(self.benchmark_data['passage'], self.benchmark_data['answer'],
    #                                          self.benchmark_data['question']):
    #         inputs = passage
    #         references.append(answer)
    #         with torch.no_grad():
    #             input_ids, attention_mask = encode_inputs['input_ids'], encode_inputs['attention_mask']
    #             input_ids = input_ids.to(self.device)
    #             outputs = self.model.generate(input_ids)
    #             decoded_outputs = self.tokenizer.decode(outputs.squeeze().tolist(), skip_special_tokens=True)
    #             predictions.append(decoded_outputs)
    #     return evaluate_metrics(predictions, references)


class QGKGGenerator:
    def __init__(self, lm_name, tokenizer, saved_qg_model, saved_kg_model, max_encoder_len, max_decoder_len):
        self.qg_model = torch.load(saved_qg_model)['state_dict']
        self.kg_model = torch.load(saved_kg_model)['state_dict']
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.qg_model.to(self.device)
        self.kg_model.to(self.device)
        self.tokenizer = tokenizer.from_pretrained(lm_name)
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

    def generate(self, p):
        with torch.no_grad():
            keyphrase_encode = self.tokenizer.encode_plus(p,
                                                          return_tensors="pt",
                                                          padding="max_length",
                                                          truncation=True,
                                                          max_length=self.max_encoder_len)
            k_input_ids = keyphrase_encode['input_ids'].to('cuda')
            g_k_encode = self.kg_model.generate(k_input_ids, max_length=self.max_decoder_len)
            g_k = self.tokenizer.decode(g_k_encode.squeeze().tolist(), skip_special_tokens=True)

            kp = g_k + ' ' + self.tokenizer.sep_token + ' ' + p
            kp_encode = self.tokenizer.encode_plus(kp,
                                                   return_tensors="pt",
                                                   padding="max_length",
                                                   truncation=True,
                                                   max_length=self.max_encoder_len)
            kp_input_ids = kp_encode['input_ids'].to('cuda')
            g_q_encode = self.qg_model.generate(kp_input_ids, max_length=self.max_decoder_len)
            g_q = self.tokenizer.decode(g_q_encode.squeeze().tolist(), skip_special_tokens=True)

            return g_k, g_q


class AGGenerator:
    def __init__(self, lm_name, tokenizer, saved_ag_model, max_encoder_len, max_decoder_len):
        self.ag_model = torch.load(saved_ag_model)['state_dict']
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.ag_model.to(self.device)
        self.tokenizer = tokenizer.from_pretrained(lm_name)
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

    def _prepare_input_for_ag(self, keyphrase, passage, question):
        input_text = keyphrase + ' {} '.format(self.tokenizer.sep_token) + passage + \
                     ' {} '.format(self.tokenizer.sep_token) + question
        encoder_inputs = self.tokenizer.encode_plus(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_encoder_len
        )
        encoder_input_ids = encoder_inputs["input_ids"][0]
        return encoder_input_ids

    def generate(self, keyphrase, passage, question):
        with torch.no_grad():
            encoder_input_ids = self._prepare_input_for_ag(keyphrase, passage, question)
            encoder_input_ids = encoder_input_ids.to('cuda')
            g_a_encode = self.ag_model(encoder_input_ids, None, None, mode='infer')
            g_a = self.tokenizer.decode(g_a_encode.squeeze().tolist(), skip_special_tokens=True)

            return g_a
