# @Time    : 2023/6/8 16:08
# @Author  : Jintao Ling
# @Email: lingjintao.su@gmail.com


class Config:
    LM = ''
    QA_DATASET = ''
    DISTRACTOR_DATASET = ''


class TaskParamsDict:
    joint_config = {
        'lm',
        'tokenizer',
        'lambda_p',
        'batch_size',
        'epochs',
        'lr',
        'vocab_size',
        'dataset',
    }

    multitask_config = {
        'lm',
        'tokenizer',
        'lambda_p',
        'batch_size',
        'epochs',
        'lr',
        'vocab_size',
        'dataset',
    }

    agtask_config = {
        'lm',
        'tokenizer',
        'lambda_p',
        'batch_size',
        'epochs',
        'lr',
        'vocab_size',
        'dataset',
    }

    qgtask_config = {
        'lm',
        'tokenizer',
        'lambda_p',
        'batch_size',
        'epochs',
        'lr',
        'vocab_size',
        'dataset',
    }

    dgtask_config = {
        'lm',
        'tokenizer',
        'max_encoder_len',
        'lambda_p',
        'batch_size',
        'epochs',
        'lr',
        'vocab_size',
        'dataset',
    }
