# @Time    : 2023/6/8 16:08
# @Author  : Jintao Ling
# @Email: lingjintao.su@gmail.com


from transformers import BartModel, BartTokenizer
from transformers import T5Model, T5Tokenizer
from transformers import ProphetNetModel, ProphetNetTokenizer


class BART:
    def __init__(self, model_name):
        self.model = BartModel.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)


class T5:
    def __init__(self, model_name):
        self.model = T5Model.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)


class ProphetNet:
    def __init__(self, model_name):
        self.model = ProphetNetModel.from_pretrained(model_name)
        self.tokenizer = ProphetNetTokenizer.from_pretrained(model_name)
