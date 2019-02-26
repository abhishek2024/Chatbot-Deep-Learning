import torch
import tensorflow as tf
import re
import numpy as np

import logging
logger = logging.getLogger(__name__)


def load_from_bert(model, n_embeddings, model_config, trainer_config):
    tf_path = trainer_config.tf_bert_model_load_from
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for name, array in zip(names, arrays):
        origin_name = name

        name = name.replace('bert/embeddings/word_embeddings', 'embeddings')
        name = name.replace('bert/embeddings/LayerNorm', 'embed_norm')
        name = name.replace('bert/embeddings/position_embeddings', 'pos_embeddings')
        name = name.replace('bert/embeddings/token_type_embeddings', 'type_embeddings')

        name = name.replace('bert/encoder/layer', 'layers')
        name = name.replace('attention/output/LayerNorm', 'attn_norm')
        name = name.replace('attention', 'attn')
        name = name.replace('attn/output/dense', 'attn/out_proj')
        name = name.replace('kernel', 'weight')
        name = name.replace('gamma', 'weight')
        name = name.replace('beta', 'bias')
        name = name.replace('output_bias', 'bias')
        name = name.replace('output_weights', 'weight')
        name = name.replace('key/bias', 'self_attn_key_bias')
        name = name.replace('key/weight', 'self_attn_key_weight')
        name = name.replace('query/bias', 'self_attn_query_bias')
        name = name.replace('query/weight', 'self_attn_query_weight')
        name = name.replace('value/bias', 'self_attn_value_bias')
        name = name.replace('value/weight', 'self_attn_value_weight')
        name = name.replace('self/', '')
        name = name.replace('intermediate/dense', 'ff/layer_1')
        name = name.replace('output/dense', 'ff/layer_2')
        name = name.replace('output/LayerNorm', 'ff_norm')

        splitted_name = name.split('/')

        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m",
                     "AdamWeightDecayOptimizer",
                     "AdamWeightDecayOptimizer_1",
                     "global_step", "cls",
                     "bert"] for n in splitted_name):
            # logger.info("Skipping {}".format(origin_name))
            continue

        pointer = model
        for m_name in splitted_name:
            if re.fullmatch(r'layers_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            try:
                if 'self_attn_' in l[0]:
                    pointer = getattr(pointer, 'qkv_proj')
                    l[0] = l[0].split('_')[-1]  # for self_attn_(key/query/value)_(weight/bias)
                pointer = getattr(pointer, l[0])
                if len(l) >= 2:
                    num = int(l[1])
                    pointer = pointer[num]
            except Exception as ex:
                logger.error(ex)
                logger.error(m_name)
                logger.error(l)
                logger.error(name)
                logger.error(origin_name)
                logger.error(f'array {array.shape}')
                return

        if 'kernel' in origin_name:
            array = np.transpose(array)
        if 'embeddings' in name:
            pointer = getattr(pointer, 'weight')

        if name == 'embeddings':
            array = array[:n_embeddings]  # slicing of embeddings
        if name == 'type_embeddings':
            mean_type_emb = array.mean(axis=0)
            new_array = np.stack([mean_type_emb]*model_config.type_vocab_size)
            start_index = model_config.type_vocab_size//2 - array.shape[0]//2
            new_array[start_index:start_index+array.shape[0]] = array
            array = new_array

        if pointer.shape != array.shape and not ('self_attn_' in name):
            logger.info(m_name)
            logger.info(l)
            logger.info(name)
            logger.info(origin_name)
            logger.info(f'pointer {pointer.shape}')
            logger.info(f'array {array.shape}')
            assert False
        if 'self_attn_' in name:
            if 'query' in name:
                shift_index = 0
            elif 'key' in name:
                shift_index = 1
            elif 'value' in name:
                shift_index = 2
            else:
                assert False
            dim1 = array.shape[0]
            pointer.data[dim1*(shift_index):dim1*(shift_index+1)] = torch.from_numpy(array)
        else:
            pointer.data = torch.from_numpy(array)
