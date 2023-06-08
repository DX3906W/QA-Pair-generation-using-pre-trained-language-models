# @Time    : 2023/6/8 16:08
# @Author  : Jintao Ling
# @Email: lingjintao.su@gmail.com


import torch

from transformers import T5Model, ProphetNetModel, BartModel
from transformers import T5ForConditionalGeneration, ProphetNetForConditionalGeneration, BartForConditionalGeneration
from transformers import T5Tokenizer, ProphetNetTokenizer, BartTokenizer


device = 'cuda'
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = torch.load('saved_models/pipeline/t5-base/answer_4.pth.tar')['state_dict']
input_encode = tokenizer.encode_plus(
        'A modern computer can be defined as a machine that stores and manipulates information under the control of a changable program.',
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=512
        )
input_ids = input_encode.input_ids
input_ids = input_ids.to(device)
model.to(device)
predictions = model.generate(input_ids)
p = tokenizer.decode(predictions.squeeze().tolist(), skip_special_tokens=True)
print(p)

