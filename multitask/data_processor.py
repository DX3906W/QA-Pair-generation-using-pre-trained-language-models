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
        mask_p = p[:answer_index] + ' <mask> ' + p[answer_index:answer_index+len(a)] + ' <mask> ' + p[answer_index+len(a):] 
        encoder_inputs = self.tokenizer.encode_plus(p,
                                        return_tensors="pt",
                                        padding="max_length",
                                        truncation=True,
                                        max_length=self.max_encoder_len)
        mask_inputs = self.tokenizer.encode_plus(mask_p,
                                     return_tensors="pt",
                                     padding="max_length",
                                     truncation=True,
                                     max_length=self.max_encoder_len)
        # print((mask_inputs["input_ids"][0] == self.tokenizer.unk_token_id).nonzero(as_tuple=True)[0])
        start_idx, end_idx = (mask_inputs["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        # start_idx -= 1
        end_idx -= 1
        decoder_inputs = self.tokenizer(q,
                                        return_tensors="pt",
                                        padding="max_length",
                                        truncation=True,
                                        max_length=self.max_decoder_len)
        
        encoder_input_ids = encoder_inputs["input_ids"][0]
        encoder_attention_mask = encoder_inputs["attention_mask"][0]

        decoder_input_ids = decoder_inputs["input_ids"][0]
        decoder_attention_mask = decoder_inputs["attention_mask"][0][:-1]
        decoder_attention_mask[0] = 0

        decoder_out = decoder_inputs["input_ids"][0][1:]
        p_tokenized = self.tokenizer.tokenize(p)
        # print(self.tokenizer.decode(encoder_input_ids))
        # print(self.tokenizer.decode(decoder_input_ids))
        # print(self.tokenizer.decode(decoder_out))
        # print(a)
        # print(self.tokenizer.decode(encoder_input_ids[start_idx:end_idx]))
        return encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask, \
            start_idx, end_idx, decoder_out
