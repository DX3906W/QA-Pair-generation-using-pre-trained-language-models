import torch

import torch.nn as nn

from transformers import T5ForConditionalGeneration


class QuestionGenerationModel(nn.Module):
    def __init__(self, lm, lm_name):
        super().__init__()
        self.qg_model = lm.from_pretrained(lm_name)
    
    def _decode(self, logits):
        return torch.argmax(logits, dim=2)

    def forward(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids, mode='train'):
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
                                         output_hidden_states=True, )

            decoder_last_hidden_state = torch.stack(res["decoder_hidden_states"][-1]).squeeze(-2)
            # print(len(res['decoder_hidden_states']))
            # print(torch.stack(res['decoder_hidden_states'][-1]).shape)
            decoder_loss = None
            decoder_out = res["sequences"]

        return decoder_last_hidden_state, decoder_loss, decoder_out


class KeyphraseGenerationModel(nn.Module):
    def __init__(self, lm, lm_name):
        super().__init__()
        self.kg_model = lm.from_pretrained(lm_name)
        self.embedding_layer = self.kg_model.get_input_embeddings()

    def _decode(self, logits):
        # dim(logits) [batch_size, seq_len, vocab_size]
        return torch.argmax(logits, dim=2)

    def forward(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids,
                question_hidden_state, mode='train'):
        inputs_embeds = self.embedding_layer(encoder_input_ids)
        if question_hidden_state is not None:
            # dim: [batch_size, seq_len, embed_dim]
            # inputs_embeds = torch.cat([question_hidden_state, inputs_embeds], dim=2)
            inputs_embeds = torch.add(question_hidden_state, inputs_embeds)

        if mode == 'train':
            res = self.kg_model(inputs_embeds=inputs_embeds,
                                attention_mask=encoder_attention_mask,
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
                                         max_length=128)
            decoder_loss = None
            decoder_out = res["sequences"]

        return decoder_loss, decoder_out


class AnswerGenerationModel(nn.Module):
    def __init__(self, lm, lm_name):
        super().__init__()
        self.ag_model = lm.from_pretrained(lm_name)

    def forward(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids, mode='train'):
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
                                         output_hidden_states=True, )
            decoder_loss = None
            decoder_out = res["sequences"]

        return decoder_loss, decoder_out
