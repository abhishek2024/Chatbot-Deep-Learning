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

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download_decompress


_DEFAULT_URL = 'http://lnsigo.mipt.ru/export/kudago/kudago_pack.tar.gz'


@register('kg_data')
class KudaGoData(Component):
    def __init__(self, url=_DEFAULT_URL, data_path='kudago', *args, **kwargs):
        data_path = expand_path(data_path)
        download_decompress(url, data_path)
        self.data = {}
        for p in data_path.iterdir():
            name = p.with_suffix('').name
            with p.open() as f:
                if p.suffix.lower() == '.json':
                    self.data[name] = json.load(f)
                elif p.suffix.lower() == '.jsonl':
                    self.data[name] = [json.loads(line) for line in f]
                else:
                    raise RuntimeError(f'unknown file suffix for {p.name}')

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
