import argparse
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
            'qgkgtask': QGKGTrainer,
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
            'batch_size': 12,
            'epochs': 5,
            'lr': 1e-5,
            'vocab_size': 50265,
            'embed_dim': 768,
            'num_heads': 12,
            'dataset': 'processed_squad',
            # 'dataset': 'race',
            'max_encoder_len': 256,
            'max_decoder_len': 128,
            # 'saved_model': None,
            # 'saved_model': './saved_models/pipeline/microsoft/prophetnet-large-uncased/question_2.pth.tar',
            # 'saved_model': './saved_models/multitask/microsoft/prophetnet-large-uncased/multi_0.pth.tar'
        }
        if task_name == 'qgtask':
            task_config['generation_task'] = 'question'
        else:
            task_config['generation_task'] = 'answer'

        self.task = self.task_names.get(task_name)(**task_config)

        if self.task is None:
            return
        parser = argparse.ArgumentParser()
        parser.add_argument('--world-size', default=2, type=int, help='number of distributed processes')
        parser.add_argument('--local_rank', type=int, help='rank of distributed processes')
        gpus_args = parser.parse_args()
        print('trainer', gpus_args)

        self.task.train()

    def analyze_file_name(self, file_name):
        return file_name.split('/')[-1].split('.')[0]

    def test_pipeline(self, lm_type, lm_name, saved_ag_model, saved_qg_model, dg_lm_type, dg_lm_name, saved_dg_model,
                      max_encoder_len,
                      max_decoder_len):
        tokenizer = self.tokenizers.get(lm_type)
        qa_file_name = self.analyze_file_name(saved_ag_model) + '_' + self.analyze_file_name(saved_qg_model) \
                       + '_d_' + self.analyze_file_name(saved_dg_model) + '_' + self.analyze_file_name(lm_name)
        param_dict = {
            'lm_name': lm_name,
            'tokenizer': tokenizer,
            'saved_ag_model': saved_ag_model,
            'saved_qg_model': saved_qg_model,
            'max_encoder_len': max_encoder_len,
            'max_decoder_len': max_decoder_len,
        }
        dg_tokenizer = self.tokenizers.get(dg_lm_type)
        generator = PipelineGenerator(**param_dict)
        dg_param_dict = {
            'lm_name': dg_lm_name,
            'tokenizer': dg_tokenizer,
            'saved_dg_model': saved_dg_model,
            'max_encoder_len': 512,
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

    def test_multitask(self, lm_type, lm_name, saved_model, dg_lm_type, dg_lm_name, saved_dg_model, max_encoder_len,
                       max_decoder_len):
        tokenizer = self.tokenizers.get(lm_type)
        qa_file_name = self.analyze_file_name(saved_model) + '_d_' + self.analyze_file_name(saved_dg_model) + '_' + \
                       lm_name.split('/')[-1]
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

    def test_qgkg(self, lm_type, lm_name, saved_qg_model, saved_kg_model, max_encoder_len, max_decoder_len):
        qa_file_name = self.analyze_file_name(saved_qg_model) + '_' + self.analyze_file_name(
            saved_kg_model) + self.analyze_file_name(lm_name)
        tokenizer = self.tokenizers.get(lm_type)
        qgkg_param_dict = {
            'lm_name': lm_name,
            'tokenizer': tokenizer,
            'max_encoder_len': max_encoder_len,
            'max_decoder_len': max_decoder_len,
            'saved_qg_model': saved_qg_model,
            'saved_kg_model': saved_kg_model,
        }
        qgkg_generator = QGKGGenerator(**qgkg_param_dict)
        benchmark_data = BenchmarkLoader().load_data('python_programming.json')
        predictions = []
        references = []
        path = './benchmark_qa/joint/{lm_type}'.format(lm_type=lm_type)
        if not os.path.exists(path):
            os.makedirs(path)
        with open('{path}/{qa_file_name}.txt'.format(path=path, qa_file_name=qa_file_name), 'w+') as f:
            for p, a, q, d in zip(benchmark_data['passage'], benchmark_data['answer'],
                                  benchmark_data['question'], benchmark_data['distractor']):
                g_k, g_q = qgkg_generator.generate(p)
                print(g_k)
                print(g_q)
                references.append(a + ' ' + q)
                predictions.append(g_k + ' ' + g_q)
                f.write(p + '\n')
                f.write(g_q + '\n')
                f.write(g_k + '\n')
                f.write('\n')
        f.close()
        print('Generated question and keyphrase evaluation: ', evaluate_metrics(predictions, references))

    def test_joint(self, lm_type, lm_name, saved_qg_model, saved_kg_model, saved_ag_model,
                   dg_lm_type, dg_lm_name, saved_dg_model, max_encoder_len, max_decoder_len):
        qa_file_name = self.analyze_file_name(saved_qg_model) + '_' + self.analyze_file_name(saved_kg_model) \
                       + '_' + self.analyze_file_name(saved_ag_model) + '_d_' + self.analyze_file_name(
            saved_dg_model) + '_' + self.analyze_file_name(lm_name)
        tokenizer = self.tokenizers.get(lm_type)
        qgkg_param_dict = {
            'lm_name': lm_name,
            'tokenizer': tokenizer,
            'max_encoder_len': max_encoder_len,
            'max_decoder_len': max_decoder_len,
            'saved_qg_model': saved_qg_model,
            'saved_kg_model': saved_kg_model,
        }
        qgkg_generator = QGKGGenerator(**qgkg_param_dict)
        ag_param_dict = {
            'lm_name': lm_name,
            'tokenizer': tokenizer,
            'max_encoder_len': max_encoder_len,
            'max_decoder_len': max_decoder_len,
            'saved_ag_model': saved_ag_model,
        }
        ag_generator = AGGenerator(**ag_param_dict)
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
        path = './benchmark_qa/joint/{lm_type}'.format(lm_type=lm_type)
        if not os.path.exists(path):
            os.makedirs(path)
        with open('{path}/{qa_file_name}.txt'.format(path=path, qa_file_name=qa_file_name), 'w+') as f:
            for p, a, q, d in zip(benchmark_data['passage'], benchmark_data['answer'],
                                  benchmark_data['question'], benchmark_data['distractor']):
                g_k, g_q = qgkg_generator.generate(p)
                g_a = ag_generator.generate(p, g_q, g_k)
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


if __name__ == "__main__":
    trainer = Trainer()
    # trainer.train('qgtask', 'prophetnet', 'microsoft/prophetnet-large-uncased')
    # trainer.train('qgkgtask', 'prophetnet', 'microsoft/prophetnet-large-uncased')
    # multi gpus setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', default=2, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='rank of distributed processes')
    gpus_args = parser.parse_args()
    print('trainer main', gpus_args)

    trainer.train('qgkgtask', 't5', 't5-base')
    # trainer.test_pipeline(lm_type='prophetnet',
    #                       lm_name='microsoft/prophetnet-large-uncased',
    #                       saved_qg_model='saved_models/pipeline/microsoft/prophetnet-large-uncased/question_3.pth.tar',
    #                       saved_ag_model='saved_models/pipeline/microsoft/prophetnet-large-uncased/answer_2.pth.tar',
    #                       dg_lm_type='t5',
    #                       dg_lm_name='t5-base',
    #                       saved_dg_model='saved_models/distractor/t5-base/1.pth.tar',
    #                       max_encoder_len=256,
    #                       max_decoder_len=128)
    # trainer.test_multitask(lm_type='prophetnet',
    #                        lm_name='microsoft/prophetnet-large-uncased',
    #                        saved_model='saved_models/multitask/microsoft/prophetnet-large-uncased/multi_0.pth.tar',
    #                        dg_lm_type='t5',
    #                        dg_lm_name='t5-base',
    #                        saved_dg_model='saved_models/distractor/t5-base/1.pth.tar',
    #                        max_encoder_len=256,
    #                        max_decoder_len=128)
    # trainer.test_joint(lm_type='t5',
    #                    lm_name='t5-base',
    #                    saved_qg_model='./saved_models/joint/t5-base/',
    #                    saved_kg_model='./saved_models/joint/t5-base/',
    #                    saved_ag_model='./saved_model/joint/t5-base/',
    #                    dg_lm_type='t5',
    #                    dg_lm_name='t5-base',
    #                    saved_dg_model='saved_models/distractor/t5-base/1.pth.tar',
    #                    max_encoder_len=256,
    #                    max_decoder_len=128)
    trainer.test_qgkg(lm_type='t5',
                      lm_name='t5-base',
                      saved_qg_model='./saved_models/joint/t5-base/qg_0_3500.pth.tar',
                      saved_kg_model='./saved_models/joint/t5-base/kg_0_3500.pth.tar',
                      max_encoder_len=256,
                      max_decoder_len=128)
