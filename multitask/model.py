import torch
import torch.nn as nn

from torch.nn import MultiheadAttention


class MultitaskModel(nn.Module):
    def __init__(self, lm, embed_dim, num_heads, vocab_size):
        super().__init__()
        self.lm = lm
        self.attention = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.fc_start = nn.Linear(embed_dim, embed_dim)
        self.fc_end = nn.Linear(embed_dim, embed_dim)
        self.vocab_size = vocab_size
        self.decoder_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids, mode='train'):
        if mode == 'train':
            res = self.lm(input_ids=encoder_input_ids,
                          attention_mask=encoder_attention_mask,
                          labels=decoder_input_ids,
                          return_dict=True,
                          output_hidden_states=True,)
            # dim [batch_size, seq_len, embed_dim]
            decoder_last_hidden_state = res["decoder_hidden_states"][-1]
            # print(torch.stack(res['decoder_hidden_states']).shape)
            encoder_last_hidden_state = res["encoder_last_hidden_state"]
            q = decoder_last_hidden_state[:, -1, :]
            # print(q.shape)
            # print(encoder_last_hidden_state.shape)
            decoder_loss = res['loss']
            decoder_out = None
        else:
            res = self.lm.generate(input_ids=encoder_input_ids,
                                   num_beams=1,
                                   return_dict_in_generate=True,
                                   output_hidden_states=True,)

            decoder_last_hidden_state = torch.stack(res["decoder_hidden_states"][-1]).squeeze(-2)
            encoder_last_hidden_state = torch.stack(res["encoder_hidden_states"])[-1]
            # print(len(res['decoder_hidden_states']))
            # print(torch.stack(res['decoder_hidden_states'][-1]).shape)
            q = decoder_last_hidden_state[-1, :]
            # print(q.shape)
            # print(encoder_last_hidden_state.shape)
            decoder_loss = None
            decoder_out = res["sequences"]

        q = torch.unsqueeze(q, 1)
        k = v = encoder_last_hidden_state

        q = torch.transpose(q, 0, 1)
        k = torch.transpose(k, 0, 1)
        v = torch.transpose(v, 0, 1)

        attention_out, attention_weight = self.attention(q, k, v)
        # dim [batch_size, embed_dim, 1]
        attention_out = attention_out.permute(1, 2, 0)

        # dim [batch_size, seq_len, embed_dim]
        start_logits = self.fc_start(encoder_last_hidden_state)
        # dim [batch_size, seq_len]
        start_logits = torch.bmm(start_logits, attention_out).squeeze(dim=-1)

        # dim [batch_size, seq_len, embed_dim]
        end_logits = self.fc_end(encoder_last_hidden_state)
        # dim [batch_size, seq_len]
        end_logits = torch.bmm(end_logits, attention_out).squeeze(dim=-1)

        return start_logits, end_logits, decoder_loss, decoder_out
