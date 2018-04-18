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
from deeppavlov.core.common.registry import register


@register('kg_filter')
class KudaGoFilter(Component):
    def __init__(self, data, n_top, data_type='places_events', *args, **kwargs):
        self.data = {item['local_id']: item for item in data[data_type]}
        self.n_top = n_top

    def __call__(self, ids, scores, tag_scores, *args, **kwargs):
        res = []
        for ids, scores, tag_scores in zip(ids, scores, tag_scores):
            ids = [item_id for item_id in ids if set(self.data[item_id]['tags']) & set(tag_scores.keys())][:self.n_top]
            data = [self.data[item_id] for item_id in ids]
            res.append([{'title': item['title'], 'local_id': item['local_id'], 'url': item['site_url']} for item in data])
        return res
