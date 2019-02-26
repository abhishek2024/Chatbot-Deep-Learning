# originally based on https://github.com/allenai/bilm-tf/blob/master/bilm/training.py

# Modifications copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
from logging import getLogger
from typing import Optional, List

#new
from attrdict import AttrDict

import numpy as np
import tensorflow as tf
from overrides import overrides

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.models.elmo.elmo2tfhub import export2hub

log = getLogger(__name__)


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

@register('transformer_chit_chat')
class TransformerChitChat(NNModel):
    """
    """

    def __init__(self,
                 options_json_path: Optional[str] = None,  # Configure by json file
                 char_cnn: Optional[dict] = None,  # Net architecture by direct params, use for overwrite a json arch.
                 bidirectional: Optional[bool] = None,
                 unroll_steps: Optional[int] = None,
                 n_tokens_vocab: Optional[int] = None,
                 lstm: Optional[dict] = None,
                 dropout: Optional[float] = None,   # Regularization
                 n_negative_samples_batch: Optional[int] = None,  # Train options
                 all_clip_norm_val: Optional[float] = None,
                 initial_accumulator_value: float = 1.0,
                 learning_rate: float = 2e-1,  # For AdagradOptimizer
                 n_gpus: int = 1,  # TODO: Add cpu supporting
                 seed: Optional[int] = None,  # Other
                 batch_size: int = 128,  # Data params
                 load_epoch_num: Optional[int] = None,
                 epoch_load_path: str = 'epochs',
                 epoch_save_path: Optional[str] = None,
                 dumps_save_path: str = 'dumps',
                 tf_hub_save_path: str = 'hubs',
                 **kwargs) -> None:

        # ================ Checking input args =================
        if not(options_json_path or (char_cnn and bidirectional and unroll_steps
                                     and n_tokens_vocab and lstm and dropout and
                                     n_negative_samples_batch and all_clip_norm_val
                                     )):
            raise Warning('Use options_json_path or/and direct params to set net architecture.')
        self.options = self._load_options(options_json_path)
        self._update_arch_options(char_cnn, bidirectional, unroll_steps, n_tokens_vocab, lstm)
        self._update_other_options(dropout, n_negative_samples_batch, all_clip_norm_val)

        # Special options
        self.options['learning_rate'] = learning_rate
        self.options['initial_accumulator_value'] = initial_accumulator_value
        self.options['seed'] = seed
        self.options['n_gpus'] = n_gpus
        self.options['batch_size'] = batch_size

        self.permanent_options = self.options

        self.train_options = {}
        self.valid_options = {'batch_size': 256, 'unroll_steps': 1, 'n_gpus': 1}

        tf.set_random_seed(seed)
        np.random.seed(seed)

        super().__init__(**kwargs)

        self.epoch_load_path = epoch_load_path

        if load_epoch_num is None:
            load_epoch_num = self._get_epoch_from(self.epoch_load_path, None)

        if epoch_save_path is None:
            self.epoch_save_path = self.epoch_load_path

        self.save_epoch_num = self._get_epoch_from(self.epoch_save_path)

        self.dumps_save_path = dumps_save_path
        self.tf_hub_save_path = tf_hub_save_path

        self._build_model(train=False, epoch=load_epoch_num)

        self.save()

    def _load_options(self, options_json_path):
        if options_json_path:
            options_json_path = expand_path(options_json_path)
            with open(options_json_path, 'r') as fin:
                options = json.load(fin)
        else:
            options = {}
        return options

    def _update_arch_options(self, char_cnn, bidirectional, unroll_steps, n_tokens_vocab, lstm):
        if char_cnn is not None:
            self.options['char_cnn'] = char_cnn
        if bidirectional is not None:
            self.options['bidirectional'] = bidirectional
        if unroll_steps is not None:
            self.options['unroll_steps'] = unroll_steps
        if n_tokens_vocab is not None:
            self.options['n_tokens_vocab'] = n_tokens_vocab
        if lstm is not None:
            self.options['lstm'] = lstm

    def _update_other_options(self, dropout, n_negative_samples_batch, all_clip_norm_val):
        if dropout is not None:
            self.options['dropout'] = dropout
        if n_negative_samples_batch is not None:
            self.options['n_negative_samples_batch'] = n_negative_samples_batch
        if all_clip_norm_val is not None:
            self.options['all_clip_norm_val'] = all_clip_norm_val

    def _get_epoch_from(self, epoch_load_path, default = 0):
        path = self.load_path
        path = path.parent / epoch_load_path
        candidates = path.resolve().glob('[0-9]*')
        candidates = list(safely_str2int(i.parts[-1]) for i in candidates
                          if safely_str2int(i.parts[-1]) is not None)
        epoch_num = max(candidates, default=default)
        return epoch_num


    def _init_session(self):
        pass


    def __call__(self, x, y, *args, **kwargs) -> List[float]:
        if len(args) != 0:
            return []

        pass
        return []

    @overrides
    def load(self, epoch: Optional[int] = None) -> None:
        """Load model parameters from self.load_path"""
        path = self.load_path
        if epoch is not None:
            path = path.parent / self.epoch_save_path / str(epoch) / path.parts[-1]
            path.resolve()
            log.info(f'[loading {epoch} epoch]')

        path = str(path)

        # Check presence of the model files
        if tf.train.checkpoint_exists(path):
            log.info(f'[loading model from {path}]')
            with self.graph.as_default():
                saver = tf.train.Saver()
                saver.restore(self.sess, path)
        else:
            log.info(f'[A checkpoint not found in  {path}]') 

    @overrides
    def save(self, epoch: Optional[int] = None) -> None:
        """Save model parameters to self.save_path"""
        path = self.save_path
        if epoch is not None:
            path = path.parent / self.epoch_save_path / str(epoch) / path.parts[-1]
            path.resolve()
            log.info(f'[saving {epoch} epoch]')

        path.parent.mkdir(parents=True, exist_ok=True)
        path = str(path)

        log.info(f'[saving model to {path}]')
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, path)

    def train_on_batch(self,
                       x_char_ids: list,
                       y_token_ids: list) -> List[float]:
        """
        This method is called by trainer to make one training step on one batch.

        Args:
            x_char_ids:  a batch of char_ids
            y_token_ids: a batch of token_ids

        Returns:
            value of loss function on batch
        """

        char_ids_batches, reversed_char_ids_batches = x_char_ids
        token_ids_batches, reversed_token_ids_batches = y_token_ids

        feed_dict = self._fill_feed_dict(char_ids_batches, reversed_char_ids_batches,
                                         token_ids_batches, reversed_token_ids_batches)

        with self.graph.as_default():
            loss, _, self.init_state_values = self.sess.run([self.loss, self.train_op, self.final_state_tensors],
                                                            feed_dict)

        return np.mean(loss)

    def _build_model(self, train: bool, epoch: Optional[int] = None, **kwargs):

        if hasattr(self, 'sess'):
            self.sess.close()

        self.options = copy.deepcopy(self.permanent_options)

        if train:
            self.options.update(self.train_options)
            self.options.update(kwargs)

            self.models, self.train_op, self.loss, self.graph = self._build_graph(tf.Graph())
        else:
            self.options.update(self.valid_options)
            self.options.update(kwargs)

            self.models, self.train_op, self.loss, self.graph = self._build_graph(tf.Graph(),
                                                                                  train=False)

        with self.graph.as_default():
            self.init_state_values, self.init_state_tensors, self.final_state_tensors =\
                self._init_session()
        self.load(epoch)



    def destroy(self) -> None:
        """
        Delete model from memory

        Returns:
            None
        """
        if hasattr(self, 'sess'):
            for k in list(self.sess.graph.get_all_collection_keys()):
                self.sess.graph.clear_collection(k)
