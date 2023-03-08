import torch
import numpy as np
import torch.nn.functional as F

from .data_processor import MultitaskDataset
from model import MultitaskModel
from transformers import AdamW
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from data_loader import *


class MultitaskTrainer:
    def __init__(self,
                 lm,
                 tokenizer,
                 lambda_p,
                 batch_size,
                 epochs,
                 lr,
                 vocab_size,
                 embed_dim,
                 num_heads,
                 dataset,
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

        self.model = MultitaskModel(self.lm, embed_dim, num_heads, vocab_size)
        self.criterion = CrossEntropyLoss(ignore_index=0)

        self.load_data()

    def load_data(self):
        if 'squad' in self.dataset:
            data = SQuADLoader().get_data()
        elif 'race' in self.dataset:
            data = RACELoader().get_data()
        else:
            data = None
        train_data, val_data, test_data = MultitaskDataset(data, self.tokenizer)
        self.train_dataloader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_data, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=True)

    def train(self):
        self.model.train()
        optimizer = AdamW(params=self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            for step, data in enumerate(self.train_dataloader):
                optimizer.zero_grad()
                batch = [d.to(self.device) for d in data]
                true_start_id, true_end_id, true_decode_id = batch[4:]
                start_logits, end_logits, decoder_out = self.model(*batch[:4])

                true_decode_id = true_decode_id.view(-1)
                decoder_out = decoder_out.view(-1, self.vocab_size)

                loss_start_idx = self.criterion(start_logits, true_start_id)
                loss_end_idx = self.criterion(end_logits, true_end_id)
                loss_decoder_idx = self.criterion(decoder_out, true_decode_id)
                loss = self.lambda_p * loss_decoder_idx + (1 - self.lambda_p) * (loss_start_idx + loss_end_idx)
                loss.backward()
                optimizer.step()

                if step % 10 == 0:
                    print("Epoch: {}  Step:{}  Loss:{}".format(epoch, step, loss.item()))

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for step, data in enumerate(self.val_dataloader):
                batch = [d.to(self.device) for d in data]
                true_start_id, true_end_id, true_decode_id = batch[4:]
                start_logits, end_logits, decoder_out = self.model(*batch[:4])

                true_decode_id = true_decode_id.view(-1)
                decoder_out = decoder_out.view(-1, self.vocab_size)

                loss_start_idx = self.criterion(start_logits, true_start_id)
                loss_end_idx = self.criterion(end_logits, true_end_id)
                loss_decoder_idx = self.criterion(decoder_out, true_decode_id)
                loss = self.lambda_p * loss_decoder_idx + (1 - self.lambda_p) * (loss_start_idx + loss_end_idx)

                if step % 10 == 0:
                    print("Step:{}  Loss:{}".format(step, loss.item()))

    def greedy_decode(self, input_text, max_encoder_length):
        self.model.eval()
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        max_question_length = 32
        max_encoder_length = 128

        encoder_inputs = self.tokenizer.encode_plus(input_text,
                                                    return_tensors="pt",
                                                    padding=True,
                                                    truncation=True,
                                                    max_length=max_encoder_length)

        encoder_input_ids = encoder_inputs["input_ids"].to(self.device)
        encoder_attention_mask = encoder_inputs["attention_mask"].to(self.device)

        decoder_input_ids = torch.tensor([[cls_id]], dtype=torch.long).to(self.device)
        decoder_attention_mask = torch.tensor([[1]], dtype=torch.long).to(self.device)

        question_ids = []
        with torch.no_grad():
            for i in range(max_question_length):
                start_logits, end_logits, decoder_out = self.model(encoder_input_ids,
                                                                   encoder_attention_mask,
                                                                   decoder_input_ids,
                                                                   decoder_attention_mask)

                values, indices = torch.topk(decoder_out, 1, dim=2)
                indice = indices[0, -1, -1].item()
                question_ids.append(indice)

                decoder_input_ids = torch.cat((decoder_input_ids, indices[:, -1, :]), dim=1).to(self.device)
                decoder_attention_mask = torch.cat((decoder_attention_mask, torch.tensor([[1]], device=self.device)), dim=1)

                if indice == sep_id or i == max_question_length - 1:
                    start_idx = torch.argmax(start_logits, dim=1)[0].item()
                    end_idx = torch.argmax(end_logits, dim=1)[0].item()
                    answer = input_text[start_idx - 1:end_idx - 1]
                    question = self.tokenizer.decode(question_ids, skip_special_tokens=True)
                    return {"question": question, "answer": answer}

    def predict(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask,
                decode_type="beam_search"):
        start_logits, end_logits, decoder_out = self.model(encoder_input_ids,
                                                           encoder_attention_mask,
                                                           decoder_input_ids,
                                                           decoder_attention_mask)
        if decode_type == "beam_search":
            decoder_out = F.log_softmax(decoder_out[:, -1, :], dim=-1)
        else:
            decoder_out = F.softmax(decoder_out[:, -1, :], dim=-1)
        return start_logits, end_logits, decoder_out

    def beam_search_decode(self, input_text, topk=3, min_len=1, min_ends=1):
        """
        Beam Search 解码，返回一条最优序列
        :param input_text: 输入文本
        :param topk: beam search 宽度
        :param min_len: 最小输出长度
        :param min_ends: 最小结束符号数目
        :return:
        """
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        max_question_length = 32
        max_encoder_length = 128

        encoder_inputs = self.tokenizer.encode_plus(input_text,
                                                    return_tensors="pt",
                                                    padding=True,
                                                    truncation=True,
                                                    max_length=max_encoder_length)

        encoder_input_ids = encoder_inputs["input_ids"].to(self.device)
        encoder_attention_mask = encoder_inputs["attention_mask"].to(self.device)

        decoder_input_ids = torch.tensor([[cls_id]], dtype=torch.long).to(self.device)
        decoder_attention_mask = torch.tensor([[1]], dtype=torch.long).to(self.device)

        # 序列输出得分
        output_scores = torch.tensor(1, dtype=torch.float).to(self.device)

        for i in range(max_question_length):
            start_logits, end_logits, scores = self.predict(encoder_input_ids,
                                                            encoder_attention_mask,
                                                            decoder_input_ids,
                                                            decoder_attention_mask)

            vocab_size = scores.shape[1]

            if i == 0:
                encoder_input_ids = encoder_input_ids[0].repeat(topk, 1)
                encoder_attention_mask = encoder_attention_mask[0].repeat(topk, 1)
                decoder_input_ids = decoder_input_ids[0].repeat(topk, 1)
                decoder_attention_mask = decoder_attention_mask[0].repeat(topk, 1)

            # 累计得分
            scores = output_scores.reshape((-1, 1)) + scores
            scores = scores.view(-1)
            values, indices = torch.topk(scores, topk)

            indices_1 = (indices // vocab_size)
            indices_2 = (indices % vocab_size).reshape((-1, 1))

            decoder_input_ids = torch.cat([decoder_input_ids[indices_1], indices_2], dim=1)
            decoder_attention_mask = torch.cat([decoder_attention_mask, torch.tensor([1]).
                                               repeat(decoder_attention_mask.shape[0], 1).to(self.device)], dim=1)

            # 更新得分
            output_scores = scores[indices]

            # 统计出现结束符号次数
            end_counts = torch.sum(decoder_input_ids == sep_id, dim=1)

            # 判断是否达到最短长度
            if decoder_input_ids.shape[1] >= min_len:
                best_one = torch.argmax(output_scores)
                # 最优路径已达到结束符号
                if end_counts[best_one] == min_ends:
                    start_idx = torch.argmax(start_logits[best_one]).item()
                    end_idx = torch.argmax(end_logits[best_one]).item()
                    answer = input_text[start_idx - 1:end_idx - 1]
                    question = self.tokenizer.decode(decoder_input_ids[best_one], skip_special_tokens=True)
                    return {"question": question, "answer": answer}
                else:
                    # 未达到结束符号序列
                    flag = (end_counts < min_ends)
                    # 有已完成序列，但是得分不是最高；删除已经完成序列
                    if not flag.all():
                        encoder_input_ids = encoder_input_ids[flag]
                        encoder_attention_mask = encoder_attention_mask[flag]
                        decoder_input_ids = decoder_input_ids[flag]
                        decoder_attention_mask = decoder_attention_mask[flag]
                        output_scores = output_scores[flag]
                        topk = flag.sum()

        # 达到设置最长长度
        best_one = torch.argmax(output_scores)
        start_idx = torch.argmax(start_logits[best_one]).item()
        end_idx = torch.argmax(end_logits[best_one])[0].item()
        answer = input_text[start_idx - 1:end_idx - 1]
        question = self.tokenizer.decode(decoder_input_ids[best_one], skip_special_tokens=True)
        return {"question": question, "answer": answer}

    def random_sample_decode(self, input_text, n, topk=None, topp=None, min_ends=1, min_len=1):
        """
        随机采样n条序列
        :param input_text: 输入文本
        :param n: 采样条数
        :param topk: 每次从概率最高的topk采样
        :param topp: 每次从概率累积达到topp的样本采样
        :param min_len: 最小输出长度
        :param min_ends: 最小结束符号数
        :return:
        """
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        max_question_length = 32
        max_encoder_length = 128

        encoder_inputs = self.tokenizer.encode_plus(input_text,
                                                    return_tensors="pt",
                                                    padding=True,
                                                    truncation=True,
                                                    max_length=max_encoder_length)

        encoder_input_ids = encoder_inputs["input_ids"].to(self.device)
        encoder_attention_mask = encoder_inputs["attention_mask"].to(self.device)

        decoder_input_ids = torch.tensor([[cls_id]], dtype=torch.long).to(self.device)
        decoder_attention_mask = torch.tensor([[1]], dtype=torch.long).to(self.device)

        questions = []
        start_ids = []
        end_ids = []

        for i in range(max_question_length):
            start_logits, end_logits, probas = self.predict(encoder_input_ids,
                                                            encoder_attention_mask,
                                                            decoder_input_ids,
                                                            decoder_attention_mask,
                                                            decode_type="random")

            if i == 0:
                encoder_input_ids = encoder_input_ids[0].repeat(n, 1)
                encoder_attention_mask = encoder_attention_mask[0].repeat(n, 1)
                decoder_input_ids = decoder_input_ids[0].repeat(n, 1)
                decoder_attention_mask = decoder_attention_mask[0].repeat(n, 1)
                probas = probas.repeat(n, 1)

            if topk is not None:
                # 取topk的索引
                k_values, k_indices = torch.topk(probas, topk, dim=-1)
                # 低版本torch不支持take_along_dim
                probas = torch.tensor(np.take_along_axis(probas.detach().cpu().numpy(),
                                                         k_indices.cpu().numpy(), axis=1)).to(self.device)
                probas /= torch.sum(probas, dim=1, keepdim=True)

            if topp is not None:
                # 降序排列，取索引
                p_indices = torch.argsort(probas, dim=1, descending=True)
                probas = torch.tensor(np.take_along_axis(probas.detach().cpu().numpy(),
                                                         p_indices.cpu().numpy(), axis=1)).to(self.device)
                # 累积概率
                cumsum_probas = torch.cumsum(probas, dim=1)
                # 标记超过topp的位置，由于超过topp的第一个位置需要保留
                # 采用roll将尾部数据移到第一个位置
                flag = torch.roll(cumsum_probas >= topp, 1, dims=1)
                flag[:, 0] = False
                # 将尾部概率较小的值置零
                probas[flag] = 0
                # 概率归一化
                probas /= torch.sum(probas, dim=1, keepdim=True)

            # 采样函数，按照概率进行采样
            sample_fun = lambda p: np.random.choice(len(p), p=p)
            sample_ids = np.apply_along_axis(sample_fun, 1, probas.detach().cpu().numpy())
            sample_ids = torch.tensor(sample_ids.reshape((-1, 1))).to(self.device)

            if topp is not None:
                sample_ids = np.take_along_axis(p_indices.detach().cpu().numpy(),
                                                sample_ids.detach().cpu().numpy(),
                                                axis=1)
            if topk is not None:
                sample_ids = np.take_along_axis(k_indices.detach().cpu().numpy(),
                                                sample_ids.detach().cpu().numpy(),
                                                axis=1)

            sample_ids = torch.tensor(sample_ids).to(self.device)

            decoder_input_ids = torch.cat([decoder_input_ids, sample_ids], dim=1)
            decoder_attention_mask = torch.cat([decoder_attention_mask,
                                                torch.tensor([1]).repeat(decoder_attention_mask.shape[0], 1).to(
                                                    self.device)],
                                               dim=1)

            end_counts = torch.sum(decoder_input_ids == sep_id, dim=1)
            if decoder_input_ids.shape[1] >= min_len:
                # 已经达到结束符号序列
                flag = (end_counts == min_ends)
                if flag.any():
                    for ids in decoder_input_ids[flag]:
                        questions.append(ids)
                    for ids in start_logits[flag]:
                        start_ids.append(ids)
                    for ids in end_logits[flag]:
                        end_ids.append(ids)

                    # 标记未完成序列
                    flag = (flag == False)
                    encoder_input_ids = encoder_input_ids[flag]
                    encoder_attention_mask = encoder_attention_mask[flag]
                    decoder_input_ids = decoder_input_ids[flag]
                    decoder_attention_mask = decoder_attention_mask[flag]

                    if len(decoder_input_ids) == 0:
                        break

        # 如果还有未完成序列，直接放入结果
        for ids in decoder_input_ids:
            questions.append(ids)
        for ids in start_logits:
            start_ids.append(ids)
        for ids in end_logits:
            end_ids.append(ids)

        start_ids = [torch.argmax(s).item() for s in start_ids]
        end_ids = [torch.argmax(s).item() for s in end_ids]
        questions = self.tokenizer.batch_decode(questions, skip_special_tokens=True)

        res = []
        for q, s, e in zip(questions, start_ids, end_ids):
            res.append({"question": q, "answer": input_text[s - 1:e - 1]})
        return res
