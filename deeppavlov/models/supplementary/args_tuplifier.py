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
from typing import Optional
from operator import itemgetter

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register


@register("args_tuplifier")
class ArgsTuplifier(Component):

    def __init__(self, sort_result=False, sort_key: Optional[int] = None, *args, **kwargs):
        self.sort_result = sort_result
        self.sort_key = sort_key

        if self.sort_result and self.sort_key is None:
            raise RuntimeError(f'{self.__class__.__name__} sort_result set to True, but sort_key is not provided.')

    def __call__(self, *args, **kwargs):

        all_results = []

        for zipped in zip(*args):
            tuples = list(zip(*zipped))
            if self.sort_result:
                tuples.sort(key=itemgetter(self.sort_key), reverse=True)
            result = []
            for i, item in enumerate(tuples, 1):
                item = list(item)
                item.insert(0, i)  # numerate tuples
                item = tuple(item)
                result.append(item)
            all_results.append(result)

        return all_results
