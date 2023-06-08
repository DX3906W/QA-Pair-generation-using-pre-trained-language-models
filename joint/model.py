# @Time    : 2023/6/8 16:08
# @Author  : Jintao Ling
# @Email: lingjintao.su@gmail.com


import torch

import torch.nn as nn

from transformers import T5ForConditionalGeneration


class QuestionGenerationModel(nn.Module):
    def __init__(self, lm, lm_name, tokenizer):
        super().__init__()
        self.qg_model = lm.from_pretrained(lm_name)
        self.qg_model.resize_token_embeddings(len(tokenizer))
    
    def _decode(self, logits):
        return torch.argmax(logits, dim=2)

    def forward(self, encoder_input_ids, encoder_attention_mask=None, decoder_input_ids=None, mode='train'):
        if mode == 'train':
            res = self.qg_model(input_ids=encoder_input_ids,
                                attention_mask=encoder_attention_mask,
                                labels=decoder_input_ids,
                                return_dict=True,
                                output_hidden_states=True, )
            # dim [batch_size, seq_len, embed_dim]
            decoder_last_hidden_state = res["decoder_hidden_states"][-1]
            # print(torch.stack(res['decoder_hidden_states']).shape)
            decoder_loss = res['loss']
            decoder_out = self._decode(res['logits'])
        else:
            res = self.qg_model.generate(input_ids=encoder_input_ids,
                                         num_beams=1,
                                         return_dict_in_generate=True,
                                         output_hidden_states=True,
                                         max_length=64)

            decoder_last_hidden_state = torch.stack(res["decoder_hidden_states"][-1]).squeeze(-2)
            # print(len(res['decoder_hidden_states']))
            # print(torch.stack(res['decoder_hidden_states'][-1]).shape)
            decoder_loss = None
            decoder_out = res["sequences"]

        return decoder_last_hidden_state, decoder_loss, decoder_out


class KeyphraseGenerationModel(nn.Module):
    def __init__(self, lm, lm_name, tokenizer):
        super().__init__()
        self.kg_model = lm.from_pretrained(lm_name)
        self.kg_model.resize_token_embeddings(len(tokenizer))
        self.embedding_layer = self.kg_model.get_input_embeddings()

    def _decode(self, logits):
        # dim(logits) [batch_size, seq_len, vocab_size]
        return torch.argmax(logits, dim=2)

    def forward(self, encoder_input_ids, encoder_attention_mask=None, decoder_input_ids=None,
                question_hidden_state=None, question_attention_mask=None, mode='train'):
        inputs_embeds = self.embedding_layer(encoder_input_ids)
        input_attention_mask = encoder_attention_mask
        if question_hidden_state is not None:
            # print(question_hidden_state.shape)
            # dim: [batch_size, seq_len, embed_dim]
            inputs_embeds = torch.cat([question_hidden_state, inputs_embeds], dim=1)
            input_attention_mask = torch.cat([question_attention_mask, encoder_attention_mask], dim=1)
            # print(inputs_embeds.shape)
            # print('encoder attention mask', encoder_attention_mask.shape)
            # print('question attention mask', question_attention_mask.shape)
            # print('input attention mask', input_attention_mask.shape)
            # inputs_embeds = torch.add(question_hidden_state, inputs_embeds)

        if mode == 'train':
            res = self.kg_model(inputs_embeds=inputs_embeds,
                                attention_mask=input_attention_mask,
                                labels=decoder_input_ids,
                                # last_decoder_hidden_state=question_hidden_state,
                                return_dict=True)
            # dim [batch_size, seq_len, embed_dim]
            decoder_loss = res['loss']
            decoder_out = self._decode(res['logits'])
        else:
            res = self.kg_model.generate(inputs_embeds=inputs_embeds,
                                         num_beams=1,
                                         return_dict_in_generate=True,
                                         # last_decoder_hidden_state=question_hidden_state,
                                         output_hidden_states=True,
                                         max_length=64)
            decoder_loss = None
            decoder_out = res["sequences"]

        return decoder_loss, decoder_out


class AnswerGenerationModel(nn.Module):
    def __init__(self, lm, lm_name, tokenizer):
        super().__init__()
        self.ag_model = lm.from_pretrained(lm_name)
        self.ag_model.resize_token_embeddings(len(tokenizer))

    def forward(self, encoder_input_ids, encoder_attention_mask=None, decoder_input_ids=None, mode='train'):
        if mode == 'train':
            res = self.ag_model(input_ids=encoder_input_ids,
                                attention_mask=encoder_attention_mask,
                                labels=decoder_input_ids,
                                return_dict=True)
            # dim [batch_size, seq_len, embed_dim]
            decoder_loss = res['loss']
            decoder_out = None
        else:
            res = self.ag_model.generate(input_ids=encoder_input_ids,
                                         num_beams=1,
                                         return_dict_in_generate=True,
                                         output_hidden_states=True,
                                         max_length=128)
            decoder_loss = None
            decoder_out = res["sequences"]

        return decoder_loss, decoder_out
