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

import shutil
from collections import OrderedDict
from logging import getLogger
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from deeppavlov.core.commands.train import get_iterator_from_config, read_data_by_config
from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.commands.utils import expand_path, parse_config
from deeppavlov.core.common.params_search import ParamsSearch

SAVE_PATH_ELEMENT_NAME = 'save_path'
# TEMP_DIR_FOR_CV = 'cv_tmp'
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


def calc_cv_score(config_, data=None, n_folds=5, tmpdir='cv_tmp', is_loo=False, del_checkpoints=False):
    config_ = parse_config(config_)
    mode = "valid"

    if data is None:
        data = read_data_by_config(config_)
    iterator = get_iterator_from_config(config_, data)

    i = 1
    cv_score = OrderedDict()
    for iterator_i in generate_train_valid(iterator, n_folds=n_folds, is_loo=is_loo):
        config, dirs_for_saved_models = change_savepath_for_model(config_, tmpdir, i)
        i += 1
        create_dirs_to_save_models(dirs_for_saved_models)
        score = train_evaluate_model_from_config(config, iterator=iterator_i)

        if del_checkpoints:
            delete_dir_for_saved_models(dirs_for_saved_models)

        if 'test' in score:
            mode = 'test'
        for key, value in score[mode].items():
            if key not in cv_score:
                cv_score[key] = []
            cv_score[key].append(value)

    for key, value in cv_score.items():
        cv_score[key] = np.mean(value)
        log.info(f"Cross-Validation -{mode}- \"{key}\" is: {cv_score[key]}")

    return cv_score
