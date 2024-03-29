# @Time    : 2023/6/8 16:08
# @Author  : Jintao Ling
# @Email: lingjintao.su@gmail.com


import codecs
import json
import os
from nltk import tokenize
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from datasets import load_dataset
from tqdm import tqdm


class SQuADLoader:
    def __init__(self):
        self.squad = load_dataset("squad")

    def load_from_local(self):
        with open('processed_squad/train_data.json', 'r') as f:
            train_json = json.load(f)

        with open('processed_squad/test_data.json', 'r') as f:
            test_json = json.load(f)

        train_contexts, train_questions, train_answers, train_answer_index = train_json.values()
        # max_train_contexts = max([len(context) for context in train_contexts])
        # sns.kdeplot([len(context) for context in train_contexts])
        # plt.show()
        test_contexts, test_questions, test_answers, test_answer_index = test_json.values()

        return list(zip(train_contexts, train_questions, train_answers, train_answer_index)), \
            list(zip(test_contexts, test_questions, test_answers, test_answer_index))

    def split_context(self, context, answer, answer_index):
        sents = tokenize.sent_tokenize(context)
        index = 0
        for sent in sents:
            next_index = index + len(sent)
            if answer in sent:
                if index <= answer_index < next_index:
                    idx = answer_index - index
                    return sent, idx
            index = next_index
            if index < len(context) and context[index] == ' ':
                index += 1

        return None, None

    def filter(self, list_data):
        temp_list = []
        for data in list_data:
            if data is not None:
                temp_list.append(data)
        return temp_list

    def load_from_online(self):
        train_squad = self.squad['train']
        test_squad = self.squad['validation']

        train_contexts = [c.strip() for c in train_squad["context"]]
        train_questions = [q.strip() for q in train_squad["question"]]
        train_answers = [a['text'][0].strip() for a in train_squad["answers"]]
        train_answer_index = [a['answer_start'][0] for a in train_squad["answers"]]

        for idx in range(len(train_contexts)):
            sub_context, answer_start = self.split_context(train_contexts[idx], train_answers[idx], train_answer_index[idx])
            if sub_context is not None and len(sub_context) >= 512:
                sub_context = None
            else:
                train_contexts[idx] = sub_context
                train_answer_index[idx] = answer_start
            if sub_context is None or answer_start is None:
                train_contexts[idx] = None
                train_answers[idx] = None
                train_questions[idx] = None
                train_answer_index[idx] = None

        test_contexts = [c.strip() for c in test_squad["context"]]
        test_questions = [q.strip() for q in test_squad["question"]]
        test_answers = [a['text'][0].strip() for a in test_squad["answers"]]
        test_answer_index = [a['answer_start'][0] for a in test_squad["answers"]]

        for idx in range(len(test_contexts)):
            sub_context, answer_start = self.split_context(test_contexts[idx], test_answers[idx], test_answer_index[idx])
            if sub_context is not None and len(sub_context) > 512:
                sub_context = None
            else:
                test_contexts[idx] = sub_context
                test_answer_index[idx] = answer_start
            if sub_context is None or answer_start is None:
                test_contexts[idx] = None
                test_answers[idx] = None
                test_questions[idx] = None
                test_answer_index[idx] = None

        train_data = {
            'context': self.filter(train_contexts),
            'questions': self.filter(train_questions),
            'answers': self.filter(train_answers),
            'answer_start': self.filter(train_answer_index),
        }
        # print(train_data['answer_start'])
        test_data = {
            'context': self.filter(test_contexts),
            'questions': self.filter(test_questions),
            'answers': self.filter(test_answers),
            'answer_start': self.filter(test_answer_index),
        }
        # print(test_data['answer_start'])
        train_json = json.dumps(train_data)
        test_json = json.dumps(test_data)

        with open('processed_squad/train_data.json', 'a') as f:
            f.write(train_json)

        with open('processed_squad/test_data.json', 'a') as f:
            f.write(test_json)

        return list(zip(train_contexts, train_questions, train_answers, train_answer_index)), \
            list(zip(test_contexts, test_questions, test_answers, test_answer_index))

    def get_data(self):
        if os.path.exists('processed_squad/train_data.json'):
            return self.load_from_local()
        else:
            return self.load_from_online()


class SQuADLoaderForJoint:
    def __init__(self):
        self.squad = load_dataset("squad")

    def load_from_local(self):
        with open('joint_squad/train_data.json', 'r') as f:
            train_json = json.load(f)

        with open('joint_squad/test_data.json', 'r') as f:
            test_json = json.load(f)

        train_contexts, train_questions, train_answers = train_json.values()
        test_contexts, test_questions, test_answers = test_json.values()

        return list(zip(train_contexts, train_questions, train_answers)), \
            list(zip(test_contexts, test_questions, test_answers))

    def get_data(self):
        train_squad = pd.DataFrame(self.squad['train'])
        train_squad = train_squad.drop(columns=['id', 'title'])
        train_squad['answers'] = train_squad['answers'].map(lambda x: x['text'][0])
        train_squad['context'] = train_squad['context'].map(lambda x: x.strip())
        train_squad_group = train_squad.groupby('context').agg(lambda x: ' <sep> '.join(x)).reset_index()

        test_squad = pd.DataFrame(self.squad['validation'])
        test_squad = test_squad.drop(columns=['id', 'title'])
        test_squad['answers'] = test_squad['answers'].map(lambda x: x['text'][0])
        test_squad['context'] = test_squad['context'].map(lambda x: x.strip())
        test_squad_group = test_squad.groupby('context').agg(lambda x: ' <sep> '.join(x)).reset_index()

        return train_squad_group, test_squad_group


class RACELoader:
    def __init__(self):
        self.race = load_dataset("race", "all")
        self.ans_2_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    def get_data(self):
        train_race = self.race['train']
        test_race = self.race['validation']

        train_contexts = [c.strip() for c in train_race["article"]]
        train_questions = [q.strip() for q in train_race["question"]]
        train_answers = [o[self.ans_2_idx[a]] for a, o in zip(train_race['answer'], train_race['options'])]

        test_contexts = [c.strip() for c in test_race["article"]]
        test_questions = [q.strip() for q in test_race["question"]]
        test_answers = [o[self.ans_2_idx[a]] for a, o in zip(test_race['answer'], test_race['options'])]

        train_placeholder = [None] * len(train_contexts)
        test_placeholder = [None] * len(test_contexts)

        return list(zip(train_contexts, train_questions, train_answers, train_placeholder)), \
            list(zip(test_contexts, test_questions, test_answers, train_placeholder))


class DGRACELoader:
    def __init__(self):
        self.file_path = 'distractor/dataset/'

    def load_data(self, file_name):
        lines = []
        # print(self.file_path + file_name)
        with codecs.open(self.file_path + file_name, "r", "utf-8") as corpus_file:
            for line in corpus_file:
                lines.append(eval(line))
        return lines


class BenchmarkLoader:
    def __init__(self):
        self.file_path = 'benchmark_dataset/'

    def load_data(self, file_name):
        with open(self.file_path + file_name, 'r') as f:
            json_dict = json.load(f)
        f.close()
        all_qa = []
        for qas in json_dict.values():
            all_qa.extend(qas)
        q = []
        a = []
        p = []
        d = []
        for qa in all_qa:
            p.append(qa['passage'])
            a.append(qa['answer'])
            q.append(qa['question'])
            d.append(qa['distractor'])

        all_qa_dict = {
            'passage': p,
            'question': q,
            'answer': a,
            'distractor': d
        }
        return all_qa_dict


if __name__ == "__main__":
    dgrace = DGRACELoader()
    for line in dgrace.load_data('race_dev_original.json'):
        print(' '.join(line['article']))
        break

