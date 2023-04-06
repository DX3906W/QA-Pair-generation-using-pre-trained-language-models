import os
import torch

from data_loader import BenchmarkLoader
from multitask.trainer import MultitaskTrainer, MultitaskGenerator
from pipeline.trainer import AGQGTrainer, PipelineGenerator
from distractor.trainer import DGTrainer, DistractorGenerator
from joint.trainer import QGKGTrainer, AGTrainer, QGKGGenerator, AGGenerator

from transformers import T5Model, ProphetNetModel, BartModel
from transformers import T5Config, ProphetNetConfig, BartConfig
from transformers import T5ForConditionalGeneration, ProphetNetForConditionalGeneration, BartForConditionalGeneration
from transformers import T5Tokenizer, ProphetNetTokenizer, BartTokenizer
from utils import evaluate_metrics


class Trainer:
    def __init__(self):
        self.task_names = {
            'multitask': MultitaskTrainer,
            'qgtask': AGQGTrainer,
            'agtask': AGQGTrainer,
            'dgtask': DGTrainer,
            'agkgtask': QGKGTrainer,
            'j_agtask': AGTrainer
        }
        self.lms = {
            't5': T5Model,
            'prophetnet': ProphetNetModel,
            'bart': BartModel,
        }
        self.generative_lms = {
            't5': T5ForConditionalGeneration,
            'prophetnet': ProphetNetForConditionalGeneration,
            'bart': BartForConditionalGeneration,
        }
        self.lm_configs = {
            't5': T5Config,
            'prophetnet': ProphetNetConfig,
            'bart': BartConfig,
        }
        self.tokenizers = {
            't5': T5Tokenizer,
            'prophetnet': ProphetNetTokenizer,
            'bart': BartTokenizer
        }

    def train(self, task_name, lm_type, lm_name):
        self.lm = self.lms.get(lm_type)
        self.generative_lm = self.generative_lms.get(lm_type)
        self.lm_name = lm_name
        self.tokenizer = self.tokenizers.get(lm_type)

        task_config = {
            'lm': self.lm,
            'generative_lm': self.generative_lm,
            'lm_name': self.lm_name,
            'tokenizer': self.tokenizer,
            'lambda_p': 0,
            'batch_size': 8,
            'epochs': 5,
            'lr': 2e-5,
            'vocab_size': 50265,
            'embed_dim': 768,
            'num_heads': 12,
            'dataset': 'processed_squad',
            'max_encoder_len': 300,
            'max_decoder_len': 128,
            'saved_model': None,
        }
        if task_name == 'qgtask':
            task_config['generation_task'] = 'question'
        else:
            task_config['generation_task'] = 'answer'

        self.task = self.task_names.get(task_name)(**task_config)

        if self.task is None:
            return

        self.task.train()

    def analyze_file_name(self, file_name):
        return file_name.split('/')[-1].split('.')[0]

    def test_pipeline(self, lm_type, lm_name, saved_ag_model, saved_qg_model, saved_dg_model, max_encoder_len,
                      max_decoder_len):
        tokenizer = self.tokenizers.get(lm_type)
        qa_file_name = self.analyze_file_name(saved_ag_model) + '_' + self.analyze_file_name(saved_qg_model) \
                       + 'd_' + self.analyze_file_name(saved_dg_model)
        param_dict = {
            'lm_name': lm_name,
            'tokenizer': tokenizer,
            'saved_ag_model': saved_ag_model,
            'saved_qg_model': saved_qg_model,
            'max_encoder_len': max_encoder_len,
            'max_decoder_len': max_decoder_len,
        }
        generator = PipelineGenerator(**param_dict)
        dg_param_dict = {
            'lm_name': lm_name,
            'tokenizer': tokenizer,
            'saved_dg_model': saved_dg_model,
            'max_encoder_len': max_encoder_len,
            'max_decoder_len': max_decoder_len,
        }
        d_generator = DistractorGenerator(**dg_param_dict)
        benchmark_data = BenchmarkLoader().load_data('python_programming.json')
        predictions = []
        references = []
        d_predictions = []
        d_references = []
        path = './benchmark_qa/pipeline/{lm_type}'.format(lm_type=lm_type)
        if not os.path.exists(path):
            os.makedirs(path)
        with open('{path}/{qa_file_name}.txt'.format(path=path, qa_file_name=qa_file_name), 'w+') as f:
            for p, a, q, d in zip(benchmark_data['passage'], benchmark_data['answer'],
                                  benchmark_data['question'], benchmark_data['distractor']):
                g_a, g_q = generator.generate(p)
                references.append(a + ' ' + q)
                predictions.append(g_a + ' ' + g_q)

                for _ in range(3):
                    g_d = d_generator.generate(p, g_q, g_a)
                    d_predictions.append(g_d)
                    d_references.append(d)

                f.write(p + '\n')
                f.write(g_q + '\n')
                f.write(g_a + '\n')
                for i in range(3, 0, -1):
                    f.write(d_predictions[-i] + '\n')
                f.write('\n')
            f.close()

        print('Generated question and answer evaluation: ', evaluate_metrics(predictions, references))
        print('Generated distractors evaluation: ', evaluate_metrics(d_predictions, d_references))

    def test_multitask(self, lm_type, lm_name, saved_model, dg_lm_type, dg_lm_name, saved_dg_model, max_encoder_len, max_decoder_len):
        tokenizer = self.tokenizers.get(lm_type)
        qa_file_name = self.analyze_file_name(saved_model) + '_d_' + self.analyze_file_name(saved_dg_model) + lm_name.split('/')[-1]
        param_dict = {
            'lm_name': lm_name,
            'tokenizer': tokenizer,
            'max_encoder_len': max_encoder_len,
            'max_decoder_len': max_decoder_len,
            'saved_model': saved_model,
        }
        generator = MultitaskGenerator(**param_dict)
        dg_tokenizer = self.tokenizers.get(dg_lm_type)
        dg_param_dict = {
            'lm_name': dg_lm_name,
            'tokenizer': dg_tokenizer,
            'saved_dg_model': saved_dg_model,
            'max_encoder_len': max_encoder_len,
            'max_decoder_len': max_decoder_len,
        }
        d_generator = DistractorGenerator(**dg_param_dict)

        benchmark_data = BenchmarkLoader().load_data('python_programming.json')
        predictions = []
        references = []
        d_predictions = []
        d_references = []
        path = './benchmark_qa/multitask/{lm_type}'.format(lm_type=lm_type)
        if not os.path.exists(path):
            os.makedirs(path)
        with open('{path}/{qa_file_name}.txt'.format(path=path, qa_file_name=qa_file_name), 'w+') as f:
            for p, a, q, d in zip(benchmark_data['passage'], benchmark_data['answer'],
                                  benchmark_data['question'], benchmark_data['distractor']):
                g_q, g_a = generator.generate(passage=p)
                # print(g_q)
                # print(g_a)
                references.append(a + ' ' + q)
                predictions.append(g_a + ' ' + g_q)

                g_d = d_generator.generate(p, g_q, g_a)
                d_predictions.append(g_d)
                d_references.append(d)
                for _ in range(3):
                    g_d = d_generator.generate(p, g_q, g_a)
                    d_predictions.append(g_d)
                    d_references.append(d)
                f.write(p + '\n')
                f.write(g_q + '\n')
                f.write(g_a + '\n')
                for i in range(3, 0, -1):
                    f.write(d_predictions[-i] + '\n')
                f.write('\n')
        f.close()

        print('Generated question and answer evaluation: ', evaluate_metrics(predictions, references))
        print('Generated distractors evaluation: ', evaluate_metrics(d_predictions, d_references))

    def test_distractor(self, lm_type, lm_name, saved_dg_model, max_encoder_len):
        lm = self.lms.get(lm_type)
        tokenizer = self.tokenizer.get(lm_type)
        param_dict = {
            'lm': lm,
            'lm_name': lm_name,
            'tokenizer': tokenizer,
            'saved_dg_model': saved_dg_model,
            'max_encoder_len': max_encoder_len,
        }
        generator = PipelineGenerator(**param_dict)
        benchmark_data = BenchmarkLoader().load_data('python_programming.json')
        predictions = []
        references = []
        for p, a, q in zip(benchmark_data['passage'], benchmark_data['answer'], benchmark_data['question']):
            g_a, g_q = generator.generate(p)
            references.append(a + ' ' + q)
            predictions.append(g_a + ' ' + g_q)
            with open('../benchmark_qa/pipeline/{lm_name}/distractor_{saved_dg_model}.txt'.format(
                    lm_name=self.lm_name, saved_dg_model=saved_dg_model), 'a') as f:
                for pre in predictions:
                    f.write(pre)
            f.close()

        print(evaluate_metrics(predictions, references))


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(task_name='agkgtask', lm_type='t5', lm_name='t5-small')
    trainer.test_multitask(lm_type='bart',
                           lm_name='facebook/bart-base',
                           saved_model='saved_models/multitask/facebook/bart-large/multi_4.pth.tar',
                           dg_lm_type='t5',
                           dg_lm_name='t5-base',
                           saved_dg_model='saved_models/distractor/t5-base/1.pth.tar',
                           max_encoder_len=256,
                           max_decoder_len=128)
