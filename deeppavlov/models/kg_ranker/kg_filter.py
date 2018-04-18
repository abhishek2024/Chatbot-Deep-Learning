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

    def __call__(self, tfidf_scores, tag_events_scores, *args, **kwargs):
        res = []
        for tfidf_scores, tag_events_scores in zip(tfidf_scores, tag_events_scores):
            tfidf_scores = self._normalize_scores(tfidf_scores)
            tag_events_scores = self._normalize_scores(tag_events_scores)
            scores = []

            for k in tfidf_scores:
                scores.append((k, tfidf_scores[k], tag_events_scores[k],))

            scores = sorted(scores, key=lambda x: x[1] + x[2], reverse=True)

            res.append([{'title': self.data[id]['title'],
                         'local_id': self.data[id]['local_id'],
                         'url': self.data[id]['site_url'],
                         'tfidf_score': tfidf_score,
                         'tag_score': tag_score,
                         } for id, tfidf_score, tag_score in scores[:self.n_top]])
        return res

    @staticmethod
    def _normalize_scores(scores):
        w = max(sum(scores.values()), 1e-06)
        for k in scores:
            scores[k] /= w
        return scores