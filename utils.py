# @Time    : 2023/6/8 16:08
# @Author  : Jintao Ling
# Email: lingjintao.su@gmail.com
# @Email: lingjintao.su@gmail.com


import evaluate


def rouge(predictions, references):
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=references, tokenizer=lambda x: x.split())
    for key in results.keys():
        results[key] *= 100
    return results


def meteor(predictions, references):
    meteor = evaluate.load("meteor")
    results = meteor.compute(predictions=predictions, references=references)['meteor'] * 100
    return results


def bleu(predictions, references, max_order):
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions, references=references, max_order=max_order)['bleu'] * 100
    return results


def evaluate_metrics(predictions, references):
    blue_score_1 = bleu(predictions, references, max_order=1)
    blue_score_2 = bleu(predictions, references, max_order=2)
    blue_score_3 = bleu(predictions, references, max_order=3)
    blue_score_4 = bleu(predictions, references, max_order=4)
    rouge_score = rouge(predictions, references)
    meteor_score = meteor(predictions, references)
    return {'blue_1': blue_score_1, 'blue_2': blue_score_2, 'blue_3': blue_score_3, 'blue_4': blue_score_4,
            'rouge': rouge_score, 'meteor': meteor_score}


def analyze_joint_output(questions, answers, separator):
    # analyze the output of the joint model
    question_list = questions.split(separator)
    answer_list = answers.split(separator)

    if not question_list[-1].endswith('?'):
        question_list = question_list[:-1]
    min_len = min(len(question_list), len(answer_list))
    question_list = question_list[:min_len]
    answer_list = answer_list[:min_len]
    assert len(question_list) == len(answer_list)

    return question_list, answer_list
