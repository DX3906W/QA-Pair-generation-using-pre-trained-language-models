import torch

from torch.utils.data import Dataset, DataLoader


class JointDataset(Dataset):
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
        question = question.replace('<sep>', self.tokenizer.cls_token)
        answer = answer.replace('<sep>', self.tokenizer.cls_token)
        return passage, question, answer
