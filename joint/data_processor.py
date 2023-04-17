import torch

from torch.utils.data import Dataset, DataLoader


class QGKGDataset(Dataset):
    def __init__(self,
                 datas,
                 tokenizer,
                 max_encoder_len=128,
                 max_decoder_len=64):
        self.datas = datas
        self.tokenizer = tokenizer

        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, index):
        passage, question, answer = self.datas.iloc[index]

        return passage, question, answer


class AGDataset(Dataset):
    def __init__(self,
                 datas,
                 tokenizer,
                 max_encoder_len=128,
                 max_decoder_len=64):
        self.datas = datas
        self.tokenizer = tokenizer

        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        p, q, a = self.datas[index]

        encoder_inputs = self.tokenizer.encode_plus(p,
                                                    return_tensors="pt",
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_encoder_len)

        decoder_inputs = self.tokenizer.encode_plus(q,
                                                    return_tensors="pt",
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_decoder_len)

        encoder_input_ids = encoder_inputs["input_ids"][0]
        encoder_attention_mask = encoder_inputs["attention_mask"][0]

        decoder_input_ids = decoder_inputs["input_ids"][0][:-1]
        decoder_output_ids = decoder_inputs["input_ids"][0][1:]

        decoder_attention_mask = decoder_inputs["attention_mask"][0][:-1]

        return encoder_input_ids, encoder_attention_mask, \
            decoder_input_ids, decoder_attention_mask, decoder_output_ids
