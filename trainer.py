from data_loader import BenchmarkLoader
from multitask.trainer import MultitaskTrainer, MultitaskGenerator
from pipeline.trainer import AGQGTrainer, PipelineGenerator
from distractor.trainer import DGTrainer

from transformers import T5Model, ProphetNetModel, BartModel
from transformers import T5ForConditionalGeneration, ProphetNetForConditionalGeneration, BartForConditionalGeneration
from transformers import T5Tokenizer, ProphetNetTokenizer, BartTokenizer
from utils import evaluate_metrics


class Trainer:
    def __init__(self,
                 task_name,
                 lm_type,
                 lm_name):
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
        self.tokenizers = {
            't5': T5Tokenizer,
            'prophetnet': ProphetNetTokenizer,
            'bart': BartTokenizer
        }
        self.lm = self.lms.get(lm_type)
        self.generative_lm = self.generative_lms.get(lm_type)
        self.lm_name = lm_name
        self.tokenizer = self.tokenizers.get(lm_type)

        task_config = {
            'lm': self.lm,
            'generative_lm': self.generative_lm,
            'lm_name': self.lm_name,
            'tokenizer': self.tokenizer,
            'lambda_p': 0.7,
            'batch_size': 4,
            'epochs': 10,
            'lr': 2e-5,
            'vocab_size': 32128,
            'embed_dim': 512,
            'num_heads': 8,
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

    def train(self):
        self.task.train()

    def test_pipeline(self, lm_type, lm_name, saved_ag_model, saved_qg_model, max_encoder_len):
        lm = self.lms.get(lm_type)
        tokenizer = self.tokenizer.get(lm_type)
        param_dict = {
            'lm': lm,
            'lm_name': lm_name,
            'tokenizer': tokenizer,
            'saved_ag_model': saved_ag_model,
            'saved_qg_model': saved_qg_model,
            'max_encoder_len': max_encoder_len,
        }
        generator = PipelineGenerator(**param_dict)
        benchmark_data = BenchmarkLoader().load_data('python_programming.txt')
        predictions = []
        references = []
        for p, a, q in zip(benchmark_data['passage'], benchmark_data['answer'], benchmark_data['question']):
            g_a, g_q = generator.generate(p)
            references.append(a + ' ' + q)
            predictions.append(g_a + ' ' + g_q)
            with open('../benchmark_qa/pipeline/{lm_name}/pipeline_{saved_ag_model}_{saved_qg_model}.txt'.format(
                    lm_name=self.lm_name, saved_ag_model=saved_ag_model, saved_qg_model=saved_qg_model), 'a') as f:
                for pre in predictions:
                    f.write(pre)
            f.close()

        print(evaluate_metrics(predictions, references))

    def test_multitask(self, lm_type, lm_name, vocab_size, embed_dim, num_heads, saved_model):
        lm = self.lms.get(lm_type)
        tokenizer = self.tokenizer.get(lm_type)
        param_dict = {
            'lm': lm,
            'lm_name': lm_name,
            'tokenizer': tokenizer,
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'max_encoder_len': 128,
            'max_decoder_len': 64,
            'saved_model': saved_model,
        }
        generator = MultitaskGenerator(**param_dict)
        benchmark_data = BenchmarkLoader().load_data('python_programming.txt')
        predictions = []
        references = []
        for p, a, q in zip(benchmark_data['passage'], benchmark_data['answer'], benchmark_data['question']):
            g_a, g_q = generator.generate('beam_search', p)
            references.append(a + ' ' + q)
            predictions.append(g_a + ' ' + g_q)
            with open('../benchmark_qa/multi_task/{lm_name}/pipeline_{saved_model}.txt'.format(
                    lm_name=self.lm_name, saved_model=saved_model, ), 'a') as f:
                for pre in predictions:
                    f.write(pre)
            f.close()

        print(evaluate_metrics(predictions, references))


if __name__ == "__main__":
    trainer = Trainer('multitask', 't5', 't5-small')
    trainer.train()
