import os
import torch
import argparse
import torch.distributed as dist
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
        self.decode_tokenizer = tokenizer.from_pretrained(lm_name)
        lm_vocab_path = './{lm_name}_vocab'.format(lm_name=lm_name)
        if not os.path.exists(lm_vocab_path):
            os.makedirs(lm_vocab_path)
        if 't5' in lm_name:
            self.tokenizer.add_special_tokens({'cls_token': '<cls>'})
            self.decode_tokenizer.add_tokens('<cls>')
        elif 'prophetnet' in lm_name:
            self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
            self.decode_tokenizer.add_tokens('[CLS]')
        elif 'bart' in lm_name:
            self.tokenizer.add_special_tokens({'cls_token': '<cls>'})
            self.decode_tokenizer.add_tokens('<cls>')
        test_text = 'this is a <cls> test'
        text_inputs = self.tokenizer.encode(test_text)
        # print(self.tokenizer.decode(text_inputs, skip_special_tokens=True))
        # print(self.decode_tokenizer.decode(text_inputs, skip_special_tokens=True))

        self.tokenizer.save_pretrained(lm_vocab_path)
        print('vocab size: ', self.tokenizer.vocab_size)
        print('special tokens: ', self.tokenizer.all_special_tokens)
        self.benchmark_data = BenchmarkLoader().load_data('python_programming.json')
        self.lm_name = lm_name

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset.lower()
        self.saved_model = saved_model

        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

        self.qg_model = QuestionGenerationModel(generative_lm, lm_name)
        self.kg_model = KeyphraseGenerationModel(generative_lm, lm_name)
        self.qg_optimizer = AdamW(params=self.qg_model.parameters(), lr=self.lr)
        self.kg_optimizer = AdamW(params=self.kg_model.parameters(), lr=self.lr)

        # if self.saved_model is not None:
        self.load_model_from_ckpt()

        self.test_sample = 'A modern computer can be defined as a machine that stores and manipulates information under the control of a  changeable program.'

    def start_train(self, rank):
        self.kg_model.train()
        self.qg_model.train()
        for epoch in range(self.epochs):
            self.train_sampler.set_epoch(epoch)
            for step, data in enumerate(self.train_dataloader):
                real_step = 1500 + step
                passage, question, answer = data
                for iter in range(5):
                    if iter == 0:
                        with torch.no_grad():
                            encoder_input_ids, encoder_attention_mask, _, _ = self._prepare_input_for_kg(passage, None, rank)
                            _, k_encode = self.kg_model(encoder_input_ids,
                                                        encoder_attention_mask,
                                                        decoder_input_ids=None,
                                                        question_hidden_state=None,
                                                        mode='infer')
                            # k_encode = k_encode.detach()
                            keyphrase = self._decode_output(k_encode)

                    self.qg_optimizer.zero_grad()
                    encoder_input_ids, encoder_attention_mask, decoder_input_ids, _ = self._prepare_input_for_qg(
                        keyphrase, passage, question, rank)
                    decoder_last_hidden_state, qg_loss, q_decoder_out = self.qg_model(
                        encoder_input_ids,
                        encoder_attention_mask,
                        decoder_input_ids,
                        mode='train')
                    qg_loss.sum().backward()
                    self.qg_optimizer.step()

                    self.kg_optimizer.zero_grad()
                    encoder_input_ids, encoder_attention_mask, decoder_input_ids, _ = self._prepare_input_for_kg(
                        passage, answer, rank)
                    kg_loss, k_decoder_out = self.kg_model(
                        encoder_input_ids,
                        encoder_attention_mask,
                        decoder_input_ids,
                        decoder_last_hidden_state.detach(),
                        mode='train')
                    keyphrase = self._decode_output(k_decoder_out)
                    kg_loss.sum().backward()
                    self.kg_optimizer.step()

                    if step % 10 == 0 and iter == 4:
                        print("Epoch: {}  Step:{}  KG Loss: {}   QG Loss: {}".format(
                            epoch, step, kg_loss.sum().item(), qg_loss.sum().item()))
                    if step % 50 == 0 and iter == 4:
                        g_q = self._decode_output(q_decoder_out)
                        print("Generated questions: ", g_q)
                        print("Generated answers: ", keyphrase)
                        print("Reference questions: ", question)
                        print("Reference answers: ", answer)
                path = './saved_models/joint/{lm_name}'.format(lm_name=self.lm_name)
                dist.barrier()
                if step > 0 and step % 200 == 0 and rank == 0:
                    folder = os.path.exists(path)
                    if not folder:
                        print('creat path')
                        os.makedirs(path)
                    torch.save({'state_dict': self.kg_model.module.state_dict(), 'optimizer': self.kg_optimizer},
                               '{path}/kg_{epoch}_{step}.pth.tar'.format(
                                   path=path, epoch=epoch, step=real_step))
                    torch.save({'state_dict': self.qg_model.module.state_dict(), 'optimizer': self.qg_optimizer},
                               '{path}/qg_{epoch}_{step}.pth.tar'.format(
                                   path=path, epoch=epoch, step=real_step))

    def load_model_from_ckpt(self):
        kg_ckpt = torch.load('./saved_models/joint/t5-base/kg_0_1500.pth.tar')
        self.kg_model = kg_ckpt['state_dict']
        self.kg_optimizer = kg_ckpt['optimizer']

        qg_ckpt = torch.load('./saved_models/joint/t5-base/qg_0_1500.pth.tar')
        self.qg_model = qg_ckpt['state_dict']
        self.qg_optimizer = qg_ckpt['optimizer']

    def load_model_from_state_dict(self, rank):
        kg_ckpt = torch.load('./saved_models/joint/t5-base/kg_0_1500.pth.tar')
        self.kg_model.load_state_dict(kg_ckpt['state_dict'])
        self.kg_optimizer.load_state_dict(kg_ckpt['optimizer'])

        qg_ckpt = torch.load('./saved_models/joint/t5-base/qg_0_1500.pth.tar')
        self.qg_model.load_state_dict(qg_ckpt['state_dict'])
        self.qg_optimizer.load_state_dict(qg_ckpt['optimizer'])

    def load_data(self):
        if 'processed_squad' in self.dataset:
            train_data, val_data = SQuADLoaderForJoint().get_data()
        elif 'race' in self.dataset:
            train_data, val_data = RACELoader().get_data()
        else:
            train_data, val_data = None, None
        train_dataset = QGKGDataset(train_data, self.tokenizer)
        val_dataset = QGKGDataset(val_data, self.tokenizer)
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

        self.train_dataloader = DataLoader(dataset=train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=2,
                                           pin_memory=True,
                                           drop_last=True,
                                           sampler=self.train_sampler)
        self.val_dataloader = DataLoader(dataset=val_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=True,
                                         num_workers=2,
                                         pin_memory=True,
                                         drop_last=True,
                                         sampler=self.val_sampler)

    def _prepare_input_for_kg(self, passage, keyphrase, rank):
        encoder_inputs = self.tokenizer(
            passage,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_encoder_len
        )
        encoder_input_ids = encoder_inputs["input_ids"].cuda(rank, non_blocking=True)
        encoder_attention_mask = encoder_inputs["attention_mask"].cuda(rank, non_blocking=True)

        if keyphrase is None:
            return encoder_input_ids, encoder_attention_mask, None, None

        decoder_inputs = self.tokenizer(
            keyphrase,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_encoder_len
        )
        decoder_input_ids = decoder_inputs["input_ids"].cuda(rank, non_blocking=True)
        decoder_attention_mask = decoder_inputs["attention_mask"].cuda(rank, non_blocking=True)

        return encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask

    def _prepare_input_for_qg(self, keyphrase, passage, question, rank):
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
        encoder_input_ids = encoder_inputs["input_ids"].cuda(rank, non_blocking=True)
        encoder_attention_mask = encoder_inputs["attention_mask"].cuda(rank, non_blocking=True)

        # question.replace('<sep>', self.tokenizer.cls_token)
        decoder_inputs = self.tokenizer(
            question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_encoder_len
        )
        decoder_input_ids = decoder_inputs["input_ids"].cuda(rank, non_blocking=True)
        decoder_attention_mask = decoder_inputs["attention_mask"].cuda(rank, non_blocking=True)

        return encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask

    def _decode_output(self, output_encode):
        batch_decoded = self.decode_tokenizer.batch_decode(output_encode, skip_special_tokens=True)
        return batch_decoded

    def train(self, gpus_args):
        torch.cuda.set_device(gpus_args.local_rank)
        dist.init_process_group(
            backend='nccl',
            world_size=gpus_args.world_size,
            rank=gpus_args.local_rank,
            init_method='env://'
        )
        self.load_data()

        self.qg_model.cuda(gpus_args.local_rank)
        self.kg_model.cuda(gpus_args.local_rank)

        self.qg_model = torch.nn.parallel.DistributedDataParallel(self.qg_model, device_ids=[gpus_args.local_rank])
        self.kg_model = torch.nn.parallel.DistributedDataParallel(self.kg_model, device_ids=[gpus_args.local_rank])
        self.start_train(gpus_args.local_rank)


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
        self.decode_tokenizer = tokenizer.from_pretrained(lm_name)
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


class QGKGGenerator:
    def __init__(self, lm_name, tokenizer, saved_qg_model, saved_kg_model, max_encoder_len, max_decoder_len):
        self.qg_model = torch.load(saved_qg_model)['state_dict']
        self.kg_model = torch.load(saved_kg_model)['state_dict']
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.qg_model.to(self.device)
        self.kg_model.to(self.device)
        self.tokenizer = tokenizer.from_pretrained(lm_name)
        self.decode_tokenizer = tokenizer.from_pretrained(lm_name)
        if 't5' in lm_name:
            self.tokenizer.add_special_tokens({'cls_token': "<cls>"})
        elif 'prophetnet' in lm_name:
            self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

    def _decode_output(self, output_encode):
        batch_decoded = self.decode_tokenizer.decode(output_encode, skip_special_tokens=True)
        return batch_decoded

    def _decode_batch_output(self, output_encode):
        batch_decoded = self.decode_tokenizer.batch_decode(output_encode, skip_special_tokens=True)
        return batch_decoded

    def generate(self, p):
        self.qg_model.eval()
        self.kg_model.eval()
        with torch.no_grad():
            passage_encode = self.tokenizer.encode_plus(p,
                                                        return_tensors="pt",
                                                        padding="max_length",
                                                        truncation=True,
                                                        max_length=self.max_encoder_len)
            k_input_ids = passage_encode['input_ids'].to(self.device)
            _, g_k_encode = self.kg_model(k_input_ids, mode='infer')
            # print('gk_encode: ', g_k_encode)
            g_k = self._decode_output(g_k_encode)
            # print(g_k)
            for i in range(5):
                kp = g_k + ' ' + self.tokenizer.cls_token + ' ' + p
                kp_encode = self.tokenizer.encode_plus(kp,
                                                       return_tensors="pt",
                                                       padding="max_length",
                                                       truncation=True,
                                                       max_length=self.max_encoder_len)
                kp_input_ids = kp_encode['input_ids'].to(self.device)
                decoder_last_hidden_state, _, g_q_encode = self.qg_model(kp_input_ids, mode='infer')
                g_q = self._decode_output(g_q_encode)
                # print(g_q)
                _, g_k_encode = self.kg_model(k_input_ids, decoder_input_ids=decoder_last_hidden_state, mode='infer')
                # print('gk_encode: ', g_k_encode)
                g_k = self._decode_output(g_k_encode)
                # print(g_k)
            return g_k, g_q

    def generate_batch(self, passages):
        self.qg_model.eval()
        self.kg_model.eval()
        with torch.no_grad():
            passage_encode = self.tokenizer.encode_plus(passages,
                                                        return_tensors="pt",
                                                        padding="max_length",
                                                        truncation=True,
                                                        max_length=self.max_encoder_len)
            k_input_ids = passage_encode['input_ids'].to(self.device)
            _, g_k_encode = self.kg_model(k_input_ids, mode='infer')
            g_k = self._decode_batch_output(g_k_encode)
            for i in range(5):
                kp = []
                for j in range(len(passages)):
                    kp.append(g_k[j] + ' ' + self.tokenizer.cls_token + ' ' + passages[j])
                kp_encode = self.tokenizer.encode_plus(kp,
                                                       return_tensors="pt",
                                                       padding="max_length",
                                                       truncation=True,
                                                       max_length=self.max_encoder_len)
                kp_input_ids = kp_encode['input_ids'].to(self.device)
                decoder_last_hidden_state, _, g_q_encode = self.qg_model(kp_input_ids, mode='infer')
                g_q = self._decode_batch_output(g_q_encode)
                # print(g_q)
                _, g_k_encode = self.kg_model(k_input_ids, decoder_input_ids=decoder_last_hidden_state, mode='infer')
                # print('gk_encode: ', g_k_encode)
                g_k = self._decode_batch_output(g_k_encode)
                # print(g_k)
            return g_k, g_q


class AGGenerator:
    def __init__(self, lm_name, tokenizer, saved_ag_model, max_encoder_len, max_decoder_len):
        self.ag_model = torch.load(saved_ag_model)['state_dict']
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.ag_model.to(self.device)
        self.tokenizer = tokenizer.from_pretrained(lm_name)
        self.decode_tokenizer = tokenizer.from_pretrained(lm_name)
        if 't5' in lm_name:
            self.tokenizer.add_special_tokens({'cls_token': "<cls>"})
        elif 'prophetnet' in lm_name:
            self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

    def _decode_output(self, output_encode):
        batch_decoded = self.decode_tokenizer.decode(output_encode, skip_special_tokens=True)
        return batch_decoded

    def _prepare_input_for_ag(self, passage, question, keyphrase):
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
            g_a = self._decode_output(g_a_encode)

            return g_a
