from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

inputs = tokenizer(['Hello, my dog is cute and beautiful and ', 'one for the output of the embeddings'],
                   padding=True,
                   return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

outputs = model.generate(input_ids, return_dict_in_generate=True, output_hidden_states=True)
# print(outputs['sequences'])

# outputs = model(input_ids, attention_mask, labels=input_ids, return_dict=True, output_hidden_states=True)

print(len(outputs['encoder_hidden_states']))
print(len(outputs['encoder_hidden_states'][0]))
print(len(outputs['encoder_hidden_states'][0][0]))
print(torch.stack(outputs['encoder_hidden_states']).shape)


