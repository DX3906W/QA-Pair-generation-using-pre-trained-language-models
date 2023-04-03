import torch

from config import Config
from utils import *
import torch.nn as nn


class AnswerGenerationModel(nn.Module):
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


class QuestionGenerationModel(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.qg_model = lm

    def forward(self, pa_input_ids, pa_attention_mask, label_ids):
        # y_ids = q_output_ids[:-1]
        # label_ids = q_output_ids[1:]

        outputs = self.qg_model(
            input_ids=pa_input_ids,
            attention_mask=pa_attention_mask,
            labels=label_ids,
        )

        return outputs
