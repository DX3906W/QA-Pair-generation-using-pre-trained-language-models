import os
import torch
import argparse
import torch.distributed as dist
from tqdm import tqdm
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

        self.qg_model = QuestionGenerationModel(generative_lm, lm_name, self.tokenizer)
        self.kg_model = KeyphraseGenerationModel(generative_lm, lm_name, self.tokenizer)
        # self.load_model_from_state_dict()
        self.test_sample = 'A modern computer can be defined as a machine that stores and manipulates information under the control of a  changeable program.'

    def start_train(self, rank):
        self.kg_model.train()
        self.qg_model.train()
        path = './saved_models/joint/{lm_name}'.format(lm_name=self.lm_name)
        folder = os.path.exists(path)
        if not folder:
            print('creat path')
            os.makedirs(path)

        for epoch in range(0, self.epochs):
            for step, data in enumerate(tqdm(self.train_dataloader)):
                real_step = step
                passage, question, answer = data
                for iter in range(3):
                    if iter == 0:
                        with torch.no_grad():
                            encoder_input_ids, encoder_attention_mask, _, _ = self._prepare_input_for_kg(passage, None,
                                                                                                         rank)
                            _, k_encode = self.kg_model(encoder_input_ids,
                                                        encoder_attention_mask,
                                                        decoder_input_ids=None,
                                                        question_hidden_state=None,
                                                        mode='infer')
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
                    question_attention_mask = torch.ones(decoder_last_hidden_state.shape[0],
                                                         decoder_last_hidden_state.shape[1]).to(f'cuda:{rank}')
                    kg_loss, k_decoder_out = self.kg_model(
                        encoder_input_ids,
                        encoder_attention_mask,
                        decoder_input_ids,
                        decoder_last_hidden_state.detach(),
                        question_attention_mask,
                        mode='train')
                    keyphrase = self._decode_output(k_decoder_out)
                    kg_loss.sum().backward()
                    self.kg_optimizer.step()

                    if step % 10 == 0 and iter == 2:
                        print("Epoch: {}  Step:{}  KG Loss: {}   QG Loss: {}".format(
                            epoch, step, kg_loss.sum().item(), qg_loss.sum().item()))
                    if step % 50 == 0 and iter == 2:
                        g_q = self._decode_output(q_decoder_out)
                        print("Generated questions: ", g_q)
                        print("Generated answers: ", keyphrase)
                        print("Reference questions: ", question)
                        print("Reference answers: ", answer)

                if step > 0 and step % 500 == 0 and rank == 0:
                    torch.save({'state_dict': self.kg_model.state_dict(), 'optimizer': self.kg_optimizer},
                               '{path}/kg_{epoch}_{step}.pth.tar'.format(
                                   path=path, epoch=epoch, step=real_step))
                    torch.save({'state_dict': self.qg_model.state_dict(), 'optimizer': self.qg_optimizer},
                               '{path}/qg_{epoch}_{step}.pth.tar'.format(
                                   path=path, epoch=epoch, step=real_step))
            torch.save({'state_dict': self.kg_model.state_dict(), 'optimizer': self.kg_optimizer},
                       '{path}/kg_{epoch}.pth.tar'.format(
                           path=path, epoch=epoch))
            torch.save({'state_dict': self.qg_model.state_dict(), 'optimizer': self.qg_optimizer},
                       '{path}/qg_{epoch}.pth.tar'.format(
                           path=path, epoch=epoch))

    def load_model_from_ckpt(self):
        kg_ckpt = torch.load('./saved_models/joint/facebook/bart-large/kg_1_4000.pth.tar', map_location='cpu')
        self.kg_model.module.load_state_dict(kg_ckpt['state_dict'].module.state_dict())
        # self.kg_optimizer = AdamW(params=self.kg_model.parameters(), lr=self.lr)

        qg_ckpt = torch.load('./saved_models/joint/facebook/bart-large/qg_1_4000.pth.tar', map_location='cpu')
        self.qg_model.module.load_state_dict(qg_ckpt['state_dict'].module.state_dict())
        # self.qg_optimizer = AdamW(params=self.qg_model.parameters(), lr=self.lr)

    def load_model_from_state_dict(self):
        kg_ckpt = torch.load('./saved_models/joint/facebook/bart-large/kg_1_4000.pth.tar', map_location='cpu')
        self.kg_model.load_state_dict(kg_ckpt['state_dict'])
        # self.kg_optimizer.load_state_dict(kg_ckpt['optimizer'])

        qg_ckpt = torch.load('./saved_models/joint/facebook/bart-large/qg_1_4000.pth.tar', map_location='cpu')
        self.qg_model.load_state_dict(qg_ckpt['state_dict'])
        # self.qg_optimizer.load_state_dict(qg_ckpt['optimizer'])

    def load_data(self, rank):
        if 'processed_squad' in self.dataset:
            train_data, val_data = SQuADLoaderForJoint().get_data()
        elif 'race' in self.dataset:
            train_data, val_data = RACELoader().get_data()
        else:
            train_data, val_data = None, None
        train_dataset = QGKGDataset(train_data, self.tokenizer)
        val_dataset = QGKGDataset(val_data, self.tokenizer)
        # self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, rank=rank)
        # self.val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=True, rank=rank)
        self.train_dataloader = DataLoader(dataset=train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           #                                   num_workers=1,
                                           #                                   pin_memory=True,
                                           drop_last=True, )
        #                                   sampler=self.train_sampler)
        self.val_dataloader = DataLoader(dataset=val_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=False,
                                         #                                 num_workers=1,
                                         #                                 pin_memory=True,
                                         drop_last=True, )
        #                                 sampler=self.val_sampler)

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

    def train(self):
        # rank = int(os.environ["RANK"])
        # world_size = int(os.environ['WORLD_SIZE'])
        # local_rank = int(os.environ['LOCAL_RANK'])
        # dist.init_process_group(backend='nccl',
        #                         world_size=world_size,
        #                         rank=rank)
        # torch.cuda.set_device(local_rank)

        self.load_data(0)
        # print('local rank', local_rank)
        # self.load_model_from_state_dict(local_rank)
        self.qg_model.cuda(0)
        self.kg_model.cuda(0)

        # self.qg_model = torch.nn.parallel.DistributedDataParallel(self.qg_model, device_ids=[local_rank], output_device=torch.device(f'cuda:{local_rank}'))
        # self.kg_model = torch.nn.parallel.DistributedDataParallel(self.kg_model, device_ids=[local_rank], output_device=torch.device(f'cuda:{local_rank}'))
        self.load_model_from_state_dict()
        self.qg_optimizer = AdamW(params=self.qg_model.parameters(), lr=self.lr)
        self.kg_optimizer = AdamW(params=self.kg_model.parameters(), lr=self.lr)

        self.start_train(0)


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
        if 't5' in lm_name:
            self.tokenizer.add_special_tokens({'cls_token': '<cls>'})
            self.decode_tokenizer.add_tokens('<cls>')
        elif 'prophetnet' in lm_name:
            self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
            self.decode_tokenizer.add_tokens('[CLS]')
        elif 'bart' in lm_name:
            self.tokenizer.add_special_tokens({'cls_token': '<cls>'})
            self.decode_tokenizer.add_tokens('<cls>')
        print('vocab size: ', self.tokenizer.vocab_size)
        print('special tokens: ', self.tokenizer.all_special_tokens)

        self.lm_name = lm_name

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset.lower()
        self.saved_model = saved_model

        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

        self.ag_model = AnswerGenerationModel(generative_lm, lm_name, self.tokenizer)
        self.ag_optimizer = AdamW(params=self.ag_model.parameters(), lr=self.lr)
        self.qgkg_generator = QGKGGenerator(
            generative_lm=generative_lm,
            lm_name=lm_name,
            tokenizer=tokenizer,
            saved_qg_model='./saved_models/joint/facebook/bart-large/qg_0.pth.tar',
            saved_kg_model='./saved_models/joint/facebook/bart-large/kg_0.pth.tar',
            max_encoder_len=max_encoder_len,
            max_decoder_len=max_decoder_len)

        if self.saved_model is not None:
            self.load_model_from_ckpt()
        self.ag_model.to(self.device)

        self.test_sample = 'A modern computer can be defined as a machine that stores and manipulates information under the control of a  changeable program.'
        self.load_data()

    def load_model_from_ckpt(self):
        ckpt = torch.load(self.saved_model)
        self.ag_model = ckpt['state_dict']
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

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
                                           drop_last=True)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def _prepare_input_for_ag(self, keyphrase, passage, question, answer):
        input_text = []
        for i in range(self.batch_size):
            input_text.append(keyphrase[i] + ' {} '.format(self.tokenizer.cls_token) + question[i] + \
                              ' {} '.format(self.tokenizer.cls_token) + passage[i])
        encoder_inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_encoder_len
        )
        encoder_input_ids = encoder_inputs["input_ids"]
        encoder_attention_mask = encoder_inputs["attention_mask"]

        decoder_inputs = self.tokenizer(
            answer,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_encoder_len
        )
        decoder_input_ids = decoder_inputs["input_ids"]
        decoder_attention_mask = decoder_inputs["attention_mask"]

        encoder_input_ids = encoder_input_ids.to(self.device)
        encoder_attention_mask = encoder_attention_mask.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)
        decoder_attention_mask = decoder_attention_mask.to(self.device)

        return encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask

    def train(self):
        self.ag_model.train()
        path = './saved_models/joint/{lm_name}'.format(lm_name=self.lm_name)
        folder = os.path.exists(path)
        if not folder:
            print('creat path')
            os.makedirs(path)

        for epoch in range(self.epochs):
            for step, data in enumerate(tqdm(self.train_dataloader)):
                passage, refer_question, answer = data
                self.ag_optimizer.zero_grad()
                keyphrase, question = self.qgkg_generator.generate_batch(passage)
                encoder_input_ids, encoder_attention_mask, decoder_input_ids, _ = self._prepare_input_for_ag(
                    keyphrase, passage, refer_question, answer)
                ag_loss, decoder_out = self.ag_model(
                    encoder_input_ids,
                    encoder_attention_mask,
                    decoder_input_ids,
                    mode='train')
                ag_loss.backward()
                self.ag_optimizer.step()

                if step % 10 == 0:
                    print("Epoch: {}  Step:{}  AG Loss: {}".format(
                        epoch, step, ag_loss.item()))
            torch.save({'state_dict': self.ag_model, 'optimizer': self.ag_optimizer},
                       '{path}/ag_{epoch}.pth.tar'.format(
                           path=path, epoch=epoch))


class QGKGGenerator:
    def __init__(self, generative_lm, lm_name, tokenizer, saved_qg_model, saved_kg_model, max_encoder_len,
                 max_decoder_len):
        self.tokenizer = tokenizer.from_pretrained(lm_name)
        self.decode_tokenizer = tokenizer.from_pretrained(lm_name)
        if 't5' in lm_name:
            self.tokenizer.add_special_tokens({'cls_token': '<cls>'})
            self.decode_tokenizer.add_tokens('<cls>')
        elif 'prophetnet' in lm_name:
            self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
            self.decode_tokenizer.add_tokens('[CLS]')
        elif 'bart' in lm_name:
            self.tokenizer.add_special_tokens({'cls_token': '<cls>'})
            self.decode_tokenizer.add_tokens('<cls>')

        self.qg_model = QuestionGenerationModel(generative_lm, lm_name, self.tokenizer)
        self.qg_model.load_state_dict(torch.load(saved_qg_model)['state_dict'])
        self.kg_model = KeyphraseGenerationModel(generative_lm, lm_name, self.tokenizer)
        self.kg_model.load_state_dict(torch.load(saved_kg_model)['state_dict'])
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.qg_model.to(self.device)
        self.kg_model.to(self.device)
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

    def _decode_output(self, output_encode):
        batch_decoded = self.decode_tokenizer.decode(output_encode.squeeze(), skip_special_tokens=True)
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
            g_k = self._decode_output(g_k_encode)
            for i in range(3):
                kp = g_k + ' ' + self.tokenizer.cls_token + ' ' + p
                kp_encode = self.tokenizer(kp,
                                           return_tensors="pt",
                                           padding="max_length",
                                           truncation=True,
                                           max_length=self.max_encoder_len)
                kp_input_ids = kp_encode['input_ids'].to(self.device)
                decoder_last_hidden_state, _, g_q_encode = self.qg_model(kp_input_ids, mode='infer')
                g_q = self._decode_output(g_q_encode)
                _, g_k_encode = self.kg_model(k_input_ids, decoder_input_ids=decoder_last_hidden_state, mode='infer')
                g_k = self._decode_output(g_k_encode)
            return g_k, g_q

    def generate_batch(self, passages):
        self.qg_model.eval()
        self.kg_model.eval()
        with torch.no_grad():
            passage_encode = self.tokenizer(passages,
                                            return_tensors="pt",
                                            padding="max_length",
                                            truncation=True,
                                            max_length=self.max_encoder_len)
            k_input_ids = passage_encode['input_ids'].to(self.device)
            _, g_k_encode = self.kg_model(k_input_ids, mode='infer')
            g_k = self._decode_batch_output(g_k_encode)
            for i in range(3):
                kp = []
                for j in range(len(passages)):
                    kp.append(g_k[j] + ' ' + self.tokenizer.cls_token + ' ' + passages[j])
                kp_encode = self.tokenizer(kp,
                                           return_tensors="pt",
                                           padding="max_length",
                                           truncation=True,
                                           max_length=self.max_encoder_len)
                kp_input_ids = kp_encode['input_ids'].to(self.device)
                decoder_last_hidden_state, _, g_q_encode = self.qg_model(kp_input_ids, mode='infer')
                g_q = self._decode_batch_output(g_q_encode)
                _, g_k_encode = self.kg_model(k_input_ids, decoder_input_ids=decoder_last_hidden_state, mode='infer')
                g_k = self._decode_batch_output(g_k_encode)
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
            self.decode_tokenizer.add_tokens('<cls>')
        elif 'prophetnet' in lm_name:
            self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
            self.decode_tokenizer.add_tokens('[CLS]')
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

    def _decode_output(self, output_encode):
        batch_decoded = self.decode_tokenizer.decode(output_encode.squeeze(), skip_special_tokens=True)
        return batch_decoded

    def _prepare_input_for_ag(self, passage, question, keyphrase):
        input_text = keyphrase + ' {} '.format(self.tokenizer.cls_token) + question + \
                     ' {} '.format(self.tokenizer.cls_token) + passage
        encoder_inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_encoder_len
        )
        encoder_input_ids = encoder_inputs["input_ids"]
        return encoder_input_ids

    def generate(self, keyphrase, passage, question):
        self.ag_model.eval()
        with torch.no_grad():
            encoder_input_ids = self._prepare_input_for_ag(keyphrase, passage, question)
            encoder_input_ids = encoder_input_ids.to('cuda')
            _, g_a_encode = self.ag_model(encoder_input_ids, None, None, mode='infer')
            g_a = self._decode_output(g_a_encode)

            return g_a
