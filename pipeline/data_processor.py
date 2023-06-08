# @Time    : 2023/6/8 16:08
# @Author  : Jintao Ling
# @Email: lingjintao.su@gmail.com


from torch.utils.data import Dataset, DataLoader


class AGDataset(Dataset):
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
        p, q, a, _ = self.datas[index]
        p_inputs = self.tokenizer.encode_plus(p,
                                              return_tensors="pt",
                                              padding="max_length",
                                              truncation=True,
                                              max_length=self.max_encoder_len)
        a_inputs = self.tokenizer.encode_plus(a,
                                              return_tensors="pt",
                                              padding="max_length",
                                              truncation=True,
                                              max_length=self.max_encoder_len)

        p_input_ids = p_inputs["input_ids"][0]
        p_attention_mask = p_inputs["attention_mask"][0]
        a_input_ids = a_inputs["input_ids"][0]

        return p_input_ids, p_attention_mask, a_input_ids


class QGDataset(Dataset):
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
        p, q, a, _ = self.datas[index]
        p_a = p + ' [SEP] ' + a
        pa_inputs = self.tokenizer.encode_plus(p_a,
                                               return_tensors="pt",
                                               padding="max_length",
                                               truncation=True,
                                               max_length=self.max_encoder_len)
        q_inputs = self.tokenizer.encode_plus(q,
                                              return_tensors="pt",
                                              padding="max_length",
                                              truncation=True,
                                              max_length=self.max_encoder_len)
        
        pa_input_ids = pa_inputs["input_ids"][0]
        pa_attention_mask = pa_inputs["attention_mask"][0]

        q_input_ids = q_inputs["input_ids"][0]

        return pa_input_ids, pa_attention_mask, q_input_ids
