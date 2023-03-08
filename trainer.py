from multitask.trainer import MultitaskTrainer
from pipeline.trainer import QGTrainer, AGTrainer
from distractor.trainer import DGTrainer

from transformers import T5Model, ProphetNetModel, BartModel
from transformers import T5ForConditionalGeneration, ProphetNetForConditionalGeneration, BartForConditionalGeneration
from transformers import T5Tokenizer, ProphetNetTokenizer, BartTokenizer


class Trainer:
    def __init__(self,
                 task_name,
                 lm_name):
        self.task_names = {
            'multitask': MultitaskTrainer,
            'qgtask': QGTrainer,
            'agtask': AGTrainer,
            'dgtask': DGTrainer,
        }
        self.lm_names = {
            't5': T5Model,
            'prophetnet': ProphetNetModel,
            'bart': BartModel,
        }
        self.genrative_lm_names = {
            't5': T5ForConditionalGeneration,
            'prophetnet': ProphetNetForConditionalGeneration,
            'bart': BartForConditionalGeneration,
        }
        self.tokenizer_names = {
            't5': T5Tokenizer,
            'prophetnet': ProphetNetTokenizer,
            'bart': BartTokenizer
        }
        self.lm = self.lm_names.get(lm_name)
        self.generative_lm = self.genrative_lm_names.get(lm_name)
        self.tokenizer = self.tokenizer_names.get(lm_name)

        task_config = {
            'lm': self.lm,
            'tokenizer': self.tokenizer,
            'lambda_p': 0.7,
            'batch_size': 8,
            'epochs': 10,
            'lr': 1e-10,
            'vocab_size': 203434,
            'dataset': 'squad',
        }

        self.task = self.task_names.get(task_name)(**task_config)

        if self.task is None:
            return

    def train(self):
        self.task.train()

    def validate(self):
        self.task.train()

    def test(self):
        self.task.train()
