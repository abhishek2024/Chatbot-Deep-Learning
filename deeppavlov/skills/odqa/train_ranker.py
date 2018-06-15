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

import argparse
from pathlib import Path
import sys
import os

from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.commands.utils import expand_path

log = get_logger(__name__)
template_config = expand_path(Path('deeppavlov/configs/odqa/basic_ranker_config.json'))

parser = argparse.ArgumentParser()
parser.add_argument("dataset_path",
                    help="Path to a dataset folder or file."
                         "Can be a local path or an URL."
                         "All dadaset files should be in utf8 encoding.",
                    type=str)
parser.add_argument("output_path", help="Path for serializable ranker data.", type=str)
parser.add_argument("lang", help="Select a language, ru (Russian) or en (English).", type=str,
                    choices={'ru', 'en'})


def train():
    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_path = args.output_path
    lang = args.lang

    reader_conf = template_config['datset_reader']


if __name__ == "__main__":
    train()
