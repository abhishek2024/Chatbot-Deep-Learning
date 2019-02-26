import argparse
from attrdict import AttrDict
from model.utils import openai_transformer_config
import glob
import hashlib
from shutil import copyfile
import pathlib
import datetime
import pprint
from tensorboardX import SummaryWriter
import sys
import random

import logging
logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logFormatter, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--signature', default='default_model')
parser.add_argument('-t', '--tie_weights', default=True, type=str2bool)
parser.add_argument('-v', '--path2bert_vocab', default='./supply/rubert_cased_L-12_H-768_A-12/std_lm_vocab.40k.txt')
parser.add_argument('-p', '--tf_bert_model_load_from', default='./supply/rubert_cased_L-12_H-768_A-12/bert_model.ckpt')
parser.add_argument('-o', '--openai_gpt_model_load_from', default='')
parser.add_argument('--openai_bpe_vocab', default=False, type=str2bool)
parser.add_argument('--bare_model', default=False, type=str2bool)
parser.add_argument('--segment_embedding', default=False, type=str2bool)
parser.add_argument('-l', '--load_last', default=True, type=str2bool)
parser.add_argument('-b', '--bert_mode', default=True, type=str2bool)
parser.add_argument('-w', '--lr_warmup', default=16000, type=int)
parser.add_argument('-m', '--lm_weight', default=0.5, type=float)
parser.add_argument('-r', '--risk_weight', default=0, type=float)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('--train_from', default='./datasets/*/*.train.txt')
parser.add_argument('--valid_from', default='./datasets/*/*.valid.txt')
args = parser.parse_args()


handler = \
    logging.FileHandler(f'logs/{str(args.signature)+ "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.txt')
handler.setLevel(logging.DEBUG)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# add the handlers to the logger
logging.getLogger('').addHandler(handler)

logger.info(' '.join(['python'] + sys.argv))
logger.info('Variable config: {}'.format(pprint.pformat(args)))
writer = SummaryWriter(f'tensorboards/{str(args.signature)}')
cfg = dotdict({'n_layers': 12, 'n_embeddings': 40477, 'n_pos_embeddings': 512, 
                   'embeddings_size': 768, 'n_heads': 12, 'dropout': 0.1,
                   'embed_dropout': 0.1, 'attn_dropout': 0.1, 'ff_dropout': 0.1})

def get_model_config():
    default_config = openai_transformer_config()
    config = AttrDict({'checkpoint_path': './checkpoints/last_checkpoint',
                       'n_layers': 12,
                       'n_pos_embeddings': 512,
                       'embeddings_size': 768,
                       'n_heads': 12,
                       'dropout': 0.1,
                       'embed_dropout': 0.1,
                       'attn_dropout': 0.1,
                       'ff_dropout': 0.1,
                       'max_seq_len': 128,
                       'sep_id_enable': True,
                       'beam_size': 3,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'annealing_topk': None,
                       'annealing': 0,
                       'length_penalty': 0.6,
                       'n_segments': True,
                       'bert_mode': True,
                       'type_vocab_size': 4,
                       'tie_weights': True,
                       })
    return config


def get_trainer_config():
    train_files = glob.glob(args.train_from)
    random.shuffle(train_files)
    valid_files = glob.glob(args.valid_from)
    config = AttrDict({'n_epochs': 1000,
                       'batch_size': 256,
                       'batch_split': 128,
                       'lr': 6.25e-5,
                       'lr_warmup': args.lr_warmup,
                       'lm_weight': args.lm_weight,
                       'risk_weight': args.risk_weight,
                       'n_jobs': 6,
                       'label_smoothing': 0.1,
                       'clip_grad': None,
                       'test_period': 1,
                       'seed': 0,
                       'device': args.device,
                       'load_last': args.load_last,
                       'bare_model': args.bare_model,
                       'tf_bert_model_load_from': args.tf_bert_model_load_from,
                       'openai_gpt_model_load_from': args.openai_gpt_model_load_from,
                       'bert_vocab': args.path2bert_vocab,
                       'last_checkpoint_path': './checkpoints/last_checkpoint',
                       'interrupt_checkpoint_path': './checkpoints/interrupt_checkpoint',
                       'train_datasets': train_files,
                       'test_datasets': valid_files,
                       'tb_writer': writer,
                       })
    return config


def get_config():
    model_config = get_model_config()
    trainer_config = get_trainer_config()
    signature = args.signature

    experiment_path = pathlib.Path('./experiments') / signature
    model_config.checkpoint_path = str(experiment_path / model_config.checkpoint_path)
    trainer_config.last_checkpoint_path = str(experiment_path / trainer_config.last_checkpoint_path)
    trainer_config.load_last = trainer_config.load_last and pathlib.Path(trainer_config.last_checkpoint_path).is_file()

    trainer_config.interrupt_checkpoint_path = str(experiment_path / trainer_config.interrupt_checkpoint_path)
    (experiment_path / 'checkpoints').mkdir(parents=True, exist_ok=True)
    return model_config, trainer_config
