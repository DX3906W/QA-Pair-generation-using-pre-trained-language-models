import torch

from data_loader import BenchmarkLoader
from multitask.trainer import MultitaskTrainer, MultitaskGenerator
from pipeline.trainer import AGQGTrainer, PipelineGenerator
from distractor.trainer import DGTrainer, DistractorGenerator

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
            'lambda_p': 0.2,
            'batch_size': 16,
            'epochs': 5,
            'lr': 2e-5,
            'vocab_size': 32100,
            'embed_dim': 768,
            'num_heads': 4,
            'dataset': 'processed_squad',
            'max_encoder_len': 512,
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

    def test_pipeline(self, lm_type, lm_name, saved_ag_model, saved_qg_model, saved_dg_model, max_encoder_len):
        tokenizer = self.tokenizers.get(lm_type)
        param_dict = {
            'lm_name': lm_name,
            'tokenizer': tokenizer,
            'saved_ag_model': saved_ag_model,
            'saved_qg_model': saved_qg_model,
            'max_encoder_len': max_encoder_len,
        }
        generator = PipelineGenerator(**param_dict)
        dg_param_dict = {
            'lm_name': lm_name,
            'tokenizer': tokenizer,
            'saved_dg_model': saved_dg_model,
            'max_encoder_len': max_encoder_len,
        }
        d_generator = DistractorGenerator(**dg_param_dict)
        benchmark_data = BenchmarkLoader().load_data('python_programming.json')
        predictions = []
        references = []
        d_predictions = []
        d_references = []

        for p, a, q, d in zip(benchmark_data['passage'], benchmark_data['answer'],
                              benchmark_data['question'], benchmark_data['distractor']):
            g_a, g_q = generator.generate(p)
            references.append(a + ' ' + q)
            predictions.append(g_a + ' ' + g_q)

            g_d = d_generator.generate(p, g_q, g_a)
            d_predictions.append(g_d)
            d_references.append(d)

            with open('../benchmark_qa/pipeline/{lm_name}/pipeline_{saved_ag_model}_{saved_qg_model}.txt'.format(
                    lm_name=self.lm_name, saved_ag_model=saved_ag_model, saved_qg_model=saved_qg_model), 'a') as f:
                f.write(p + '\n')
                f.write(g_q + '\n')
                f.write(g_a + '\n')
                f.write('\n')
            f.close()

        print('Generated question and answer evaluation: ', evaluate_metrics(predictions, references))
        print('Generated distractors evaluation: ', evaluate_metrics(d_predictions, d_references))

    def test_multitask(self, lm_type, lm_name, vocab_size, embed_dim, num_heads, saved_model,
                       saved_dg_model, max_encoder_len):
        lm = self.lms.get(lm_type)
        tokenizer = self.tokenizers.get(lm_type)
        param_dict = {
            'lm_name': lm_name,
            'tokenizer': tokenizer,
            'max_encoder_len': 128,
            'max_decoder_len': 64,
            'saved_model': saved_model,
        }
        generator = MultitaskGenerator(**param_dict)
        dg_param_dict = {
            'lm_name': lm_name,
            'tokenizer': tokenizer,
            'saved_dg_model': saved_dg_model,
            'max_encoder_len': max_encoder_len,
        }
        d_generator = DistractorGenerator(**dg_param_dict)

        benchmark_data = BenchmarkLoader().load_data('python_programming.json')
        predictions = []
        references = []
        d_predictions = []
        d_references = []

        for p, a, q, d in zip(benchmark_data['passage'], benchmark_data['answer'],
                              benchmark_data['question'], benchmark_data['distractor']):
            g_a, g_q = generator.generate('beam_search', p)
            references.append(a + ' ' + q)
            predictions.append(g_a + ' ' + g_q)

            g_d = d_generator.generate(p, g_q, g_a)
            d_predictions.append(g_d)
            d_references.append(d)

            with open('../benchmark_qa/multi_task/{lm_name}/pipeline_{saved_model}.txt'.format(
                    lm_name=self.lm_name, saved_model=saved_model, ), 'a') as f:
                f.write(p + '\n')
                f.write(g_q + '\n')
                f.write(g_a + '\n')
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
    # trainer.train('dgtask', 'prophetnet', 'microsoft/prophetnet-large-uncased')
    trainer.test_pipeline('t5',
                          't5-small',
                          'saved_models/pipeline/t5-small/answer_t5-small_0.pth.tar',
                          'saved_models/pipeline/t5-small/answer_t5-small_0.pth.tar',
                          'saved_models/pipeline/t5-small/answer_t5-small_0.pth.tar',
                          512)
    # trainer.test_multitask('t5',
    #                        't5-small',
    #                        30512,
    #                        512,
    #                        4,
    #                        'saved_models/multitask/t5-small/multi_t5-small_0.pth.tar',
    #                        'saved_models/pipeline/t5-small/answer_t5-small_0.pth.tar',
    #                        512)
