import torch

from data_loader import DGRACELoader
from torch.utils.data import Dataset


class DistractorDataset(Dataset):
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
        p = ' '.join(self.datas[index]['article'])
        q = ' '.join(self.datas[index]['question'])
        a = ' '.join(self.datas[index]['answer_text'])
        d = ' '.join(self.datas[index]['distractor'])
        
        input_text = 'Generated distractor: ' + q + ' answer: ' + a + ' context: ' + p + ' </s>'
        
        encoder_inputs = self.tokenizer.encode_plus(input_text,
                                                    return_tensors="pt",
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_encoder_len)
        decoder_inputs = self.tokenizer.encode_plus(d,
                                                    return_tensors="pt",
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_decoder_len)

        encoder_input_ids = encoder_inputs["input_ids"][0]
        encoder_attention_mask = encoder_inputs["attention_mask"][0]

        decoder_input_ids = decoder_inputs["input_ids"][0]

        return encoder_input_ids, encoder_attention_mask, decoder_input_ids
