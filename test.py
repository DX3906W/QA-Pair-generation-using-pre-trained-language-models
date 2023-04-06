from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
import torch


# model = T5ForConditionalGeneration.from_pretrained('t5-small')
# tokenizer = T5Tokenizer.from_pretrained('t5-small')

model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

inputs = tokenizer(['Hello, my dog is cute, beautiful and ', 'Hello, my cat is '],
                   padding=True,
                   return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

outputs = model.generate(input_ids, max_length=10, return_dict_in_generate=True, output_hidden_states=True)
print(outputs['sequences'])
print(torch.mul((outputs['sequences'] == 1)[:, 1:], torch.ones(2, 9)))
print(outputs['sequences'].shape)

# outputs = model(input_ids, attention_mask, labels=input_ids, return_dict=True, output_hidden_states=True)

# print(len(outputs['encoder_hidden_states']))
# print(len(outputs['encoder_hidden_states'][0]))
# print(len(outputs['encoder_hidden_states'][0][0]))
print(torch.stack(outputs['encoder_hidden_states']).shape)
print(len(outputs['decoder_hidden_states']))

x = torch.stack(tuple(torch.stack(outputs['decoder_hidden_states'][i])[-1] for i in range(len(outputs['decoder_hidden_states'])))).squeeze().transpose(0, 1)
print(torch.stack(tuple(torch.stack(outputs['decoder_hidden_states'][i])[-1] for i in range(len(outputs['decoder_hidden_states'])))).squeeze().transpose(0, 1).shape)


