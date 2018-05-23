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
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.common.registry import register


@register('nerdict')
class NERDict(Component, Serializable):
    def __init__(self, **kwargs):
        super(NERDict, self).__init__(**kwargs)
        self.cities = None

    def load(self):
        with open(self.load_path) as f:
            self.cities = {line.strip() for line in f}

    def __call__(self, tokens_batch, **kwargs):
        lines_batch = [' '.join(utt) for utt in tokens_batch]
        tags_batch = []
        for utt in tokens_batch:
            tags_batch.append([tok in self.cities for tok in utt])
        return tags_batch
