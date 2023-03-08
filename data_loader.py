import codecs
import json

from datasets import load_dataset


class SQuADLoader:
    def __init__(self):
        squad = load_dataset("squad")
        self.squad = squad.train_test_split(test_size=0.2)

    def get_data(self):
        train_squad = self.squad['train']
        test_squad = self.squad['test']

        train_contexts = [c.strip() for c in train_squad["context"]]
        train_questions = [q.strip() for q in train_squad["question"]]
        train_answers = [a.strip() for a in train_squad["answer"]]

        test_contexts = [c.strip() for c in test_squad["context"]]
        test_questions = [q.strip() for q in test_squad["question"]]
        test_answers = [a.strip() for a in test_squad["answer"]]

        return list(zip(train_contexts, train_questions, train_answers)), \
            list(zip(test_contexts, test_questions, test_answers))


class RACELoader:
    def __init__(self):
        race = load_dataset("race")
        self.race = race.train_test_split(test_size=0.2)

    def get_data(self):
        train_race = self.race['train']
        test_race = self.race['test']

        train_contexts = [c.strip() for c in train_race["context"]]
        train_questions = [q.strip() for q in train_race["question"]]
        train_answers = [a.strip() for a in train_race["answer"]]

        test_contexts = [c.strip() for c in test_race["context"]]
        test_questions = [q.strip() for q in test_race["question"]]
        test_answers = [a.strip() for a in test_race["answer"]]

        return list(zip(train_contexts, train_questions, train_answers)), \
            list(zip(test_contexts, test_questions, test_answers))


class DGRACELoader:
    def __init__(self,
                 file_path='distractor/dataset/'):
        self.file_path = file_path

    def load_data(self, file_name):
        lines = []
        with codecs.open(self.file_path + file_name, "r", "utf-8") as corpus_file:
            for line in corpus_file:
                 lines.append(eval(line))
        return lines


if __name__ == "__main__":
    dgrace = DGRACELoader()
    for line in dgrace.load_data('race_dev_original.json'):
        print(' '.join(line['article']))
        break

