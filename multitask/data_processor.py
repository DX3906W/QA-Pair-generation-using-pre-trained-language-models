import torch

from torch.utils.data import Dataset, DataLoader


class MultitaskDataset(Dataset):
    def __init__(self,
                 datas,
                 tokenizer,
                 max_encoder_len=128,
                 max_decoder_len=64):
        self.tokenizer = tokenizer
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        p, q, a, answer_index = self.datas[index]
        # print(p)
        # print(q)
        # print(a)
        # print(start_idx)
        start_idx = 0
        q_tokenized = self.tokenizer.tokenize(p)
        a_tokenized = self.tokenizer.tokenize(a)
        # print(q_tokenized)
        # print(a_tokenized)
        for i in range(len(q_tokenized)):
            if answer_index == 0:
                start_idx = i + 1
                break
            answer_index -= len(q_tokenized[i])
        # print(start_idx)
        end_idx = start_idx + len(a_tokenized)
        # print(end_idx)
        # print(q_tokenized[start_idx:end_idx])
        encoder_inputs = self.tokenizer.encode_plus(p,
                                                    return_tensors="pt",
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_encoder_len)
        q = '<pad> ' + q
        decoder_inputs = self.tokenizer.encode_plus(q,
                                                    return_tensors="pt",
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_decoder_len)

        encoder_input_ids = encoder_inputs["input_ids"][0]
        encoder_attention_mask = encoder_inputs["attention_mask"][0]

        decoder_input_ids = decoder_inputs["input_ids"][0][:-1]
        decoder_attention_mask = decoder_inputs["attention_mask"][0][:-1]
        decoder_attention_mask[0] = 0

        decoder_out = decoder_inputs["input_ids"][0][1:]

        start_idx = torch.tensor(start_idx, dtype=torch.long)
        end_idx = torch.tensor(end_idx, dtype=torch.long)

        return encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask, \
            start_idx, end_idx, decoder_out
