import torch.nn as nn


class DistractorGenerationModel(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.ag_model = lm

    def forward(self, p_input_ids, p_attention_mask, label_ids):
        outputs = self.ag_model(
            input_ids=p_input_ids,
            attention_mask=p_attention_mask,
            labels=label_ids,
        )

        return outputs


