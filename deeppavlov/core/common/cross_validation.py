# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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

import os
import shutil
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Generator

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from deeppavlov.core.commands.train import get_iterator_from_config, read_data_by_config, \
    train_evaluate_model_from_config
from deeppavlov.core.commands.utils import expand_path, parse_config
from deeppavlov.core.common.params_search import ParamsSearch
from deeppavlov.pipeline_manager.utils import get_available_gpus

SAVE_PATH_ELEMENT_NAME = 'save_path'
log = getLogger(__name__)


def change_savepath_for_model(config, tmp_dir, k):
    TEMP_DIR_FOR_CV = tmp_dir
    params_helper = ParamsSearch()
    dirs_for_saved_models = set()
    for p in params_helper.find_model_path(config, SAVE_PATH_ELEMENT_NAME):
        p.append(SAVE_PATH_ELEMENT_NAME)
        save_path = Path(params_helper.get_value_from_config(config, p))
        new_save_path = save_path.parent / TEMP_DIR_FOR_CV / f"fold_{k}" / save_path.name

        dirs_for_saved_models.add(expand_path(new_save_path.parent))

        params_helper.insert_value_or_dict_into_config(config, p, str(new_save_path))

    return config, dirs_for_saved_models


def delete_dir_for_saved_models(dirs_for_saved_models):
    for new_save_dir in dirs_for_saved_models:
        shutil.rmtree(str(new_save_dir))


def create_dirs_to_save_models(dirs_for_saved_models):
    for new_save_dir in dirs_for_saved_models:
        new_save_dir.mkdir(exist_ok=True, parents=True)


def generate_train_valid(iterator_, n_folds=5, is_loo=False):
    all_data = iterator_.data['train'] + iterator_.data['valid']

    if is_loo:
        # for Leave One Out
        for i in range(len(all_data)):
            iterator_.data['train'] = all_data.copy()
            iterator_.data['valid'] = [iterator_.data['train'].pop(i)]
            yield iterator_
    else:
        # for Cross Validation
        kf = KFold(n_splits=n_folds, shuffle=True)
        for train_index, valid_index in kf.split(all_data):
            iterator_.data['train'] = [all_data[i] for i in train_index]
            iterator_.data['valid'] = [all_data[i] for i in valid_index]
            yield iterator_


class CrossVal:
    def __init__(self,
                 config,
                 data=None,
                 mode='valid',
                 n_folds=5,
                 tmpdir='cv_tmp',
                 is_loo=False,
                 del_checkpoints=False):
        self.n_folds = n_folds
        self.folds_dir = tmpdir
        self.is_loo = is_loo
        self.mode = mode
        self.clean_checkpoint = del_checkpoints
        self.config = parse_config(config)

        if data is None:
            self.data = read_data_by_config(self.config)
        else:
            self.data = data

        self.iterator = get_iterator_from_config(self.config, self.data)
        self.cv_score = OrderedDict()

        self.parallel = self.config['crossval'].get('parallel')
        self.use_gpu = self.config['crossval'].get('use_gpu')
        self.memory_fraction = self.config['crossval'].get('memory_fraction', 1.0)
        self.available_gpu = None
        self.max_num_workers = None
        self.prepare_multiprocess()

    def prepare_multiprocess(self) -> None:
        """
        Calculates the number of workers and the set of available video cards, if gpu is used, based on init attributes.
        """
        try:
            visible_gpu = [int(q) for q in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
        except KeyError:
            visible_gpu = None

        if self.use_gpu:
            if isinstance(self.use_gpu, (list, int)):
                if visible_gpu:
                    self.use_gpu = list(set(self.use_gpu) & set(visible_gpu))

                if len(self.use_gpu) == 0:
                    raise ValueError("GPU numbers in 'use_gpu' and 'CUDA_VISIBLE_DEVICES' "
                                     "has not intersections".format(set(visible_gpu)))

                self.available_gpu = get_available_gpus(gpu_select=self.use_gpu, gpu_fraction=self.memory_fraction)
            elif self.use_gpu == "all":
                self.available_gpu = get_available_gpus(gpu_select=visible_gpu, gpu_fraction=self.memory_fraction)

            if len(self.available_gpu) == 0:
                raise ValueError("All selected GPU with numbers: ({}), are busy.".format(set(self.use_gpu)))
            elif len(self.available_gpu) < len(self.use_gpu):
                print("PipelineManagerWarning: 'CUDA_VISIBLE_DEVICES' = ({0}), "
                      "but only {1} are available.".format(self.use_gpu, self.available_gpu))

            self.max_num_workers = len(self.available_gpu)
            if self.n_folds < self.max_num_workers:
                self.max_num_workers = self.n_folds

    def gpu_gen(self) -> Generator:
        """Create generator that returning tuple of args fore self.train_pipe method."""
        for i, iterator_i in enumerate(generate_train_valid(self.iterator, n_folds=self.n_folds, is_loo=self.is_loo)):
            gpu_ind = i % len(self.available_gpu)
            yield (self.config, iterator_i, self.folds_dir, i, self.available_gpu[gpu_ind])

    @staticmethod
    def k_train(base_config, iterator_i, folds_dir, k, gpu_ind):
        # modify project environment
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ind)
        config_, dirs_for_saved_models = change_savepath_for_model(base_config, folds_dir, k + 1)
        create_dirs_to_save_models(dirs_for_saved_models)
        score = train_evaluate_model_from_config(config_, iterator=iterator_i)
        return score

    def run(self):
        if self.parallel:
            with ProcessPoolExecutor(self.max_num_workers) as executor:
                futures = [executor.submit(self.k_train, *args) for args in self.gpu_gen()]
                for res in tqdm(as_completed(futures), total=len(futures)):
                    score = res.result()
                    if 'test' in score:
                        self.mode = 'test'
                    for key, value in score[self.mode].items():
                        if key not in self.cv_score:
                            self.cv_score[key] = []
                        self.cv_score[key].append(value)

            for key, value in self.cv_score.items():
                self.cv_score[key] = np.mean(value)
                log.info(f"Cross-Validation -{self.mode}- \"{key}\" is: {self.cv_score[key]}")
                print(f"Cross-Validation -{self.mode}- \"{key}\" is: {self.cv_score[key]}")
        else:
            for i, iterator_i in enumerate(generate_train_valid(self.iterator,
                                                                n_folds=self.n_folds,
                                                                is_loo=self.is_loo)):
                config_, dirs_for_saved_models = change_savepath_for_model(deepcopy(self.config), self.folds_dir, i)
                create_dirs_to_save_models(dirs_for_saved_models)
                score = train_evaluate_model_from_config(config_, iterator=iterator_i)

                if self.clean_checkpoint:
                    delete_dir_for_saved_models(dirs_for_saved_models)

                if 'test' in score:
                    self.mode = 'test'
                for key, value in score[self.mode].items():
                    if key not in self.cv_score:
                        self.cv_score[key] = []
                    self.cv_score[key].append(value)

            for key, value in self.cv_score.items():
                self.cv_score[key] = np.mean(value)
                log.info(f"Cross-Validation -{self.mode}- \"{key}\" is: {self.cv_score[key]}")
                print(f"Cross-Validation -{self.mode}- \"{key}\" is: {self.cv_score[key]}")
