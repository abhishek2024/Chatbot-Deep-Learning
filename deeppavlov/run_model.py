"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import json

from deeppavlov.core.commands.train import train_evaluate_model_from_config, build_model_from_config
from deeppavlov.core.commands.infer import interact_model
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.file import read_json

# config_path = 'configs/odqa/en_ranker_news.json'
# config_path = 'configs/odqa/en_ranker_drones.json'
# config_path = 'configs/odqa/en_ranker_sber.json'
# config_path = 'configs/odqa/en_ranker_wiki.json'
# config_path = 'configs/odqa/ru_ranker_corp.json'
# config_path = 'configs/odqa/ru_ranker_drones.json'


# config_path ="configs/odqa/en_ranker_tfidf_train.json"
# train_evaluate_model_from_config(config_path)

# config_path = '/home/gureenkova/tmp/pycharm_project_196/deeppavlov/configs/odqa/en_ranker_ensemble_drones.json'
# config_path = '/media/olga/Data/projects/iPavlov/DeepPavlov/deeppavlov/configs/odqa/en_ranker_tfidf_drones.json'
config_path = '/media/olga/Data/projects/iPavlov/DeepPavlov/deeppavlov/configs/odqa/en_ranker_ensemble_drones.json'

# interact_model(config_path)
#
model = build_model_from_config(read_json(config_path))
data = ["What corporation manufactures drones in Russia?", "Maximum speed of drones?", "What parts are the most significant?"]
result = model(data)
print(result)
print('Done!')






