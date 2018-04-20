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
    def __init__(self, data, n_top, data_type='places_events',
                 tfidf_weight=1.0, tag_weight=1.0, cluster_weight=1.0, *args, **kwargs):
        self.data = {item['local_id']: item for item in data[data_type]}
        self.n_top = n_top
        self.tfidf_weight = tfidf_weight
        self.tag_weight = tag_weight
        self.cluster_weight = cluster_weight

    def _filter_time_span(self, event_id, time_span):
        ev = self.data[event_id]
        if time_span is None or ev['data_type'] == 'places':
            return True
        start, stop = [int(t.strftime('%s')) for t in time_span]
        if ev['data_type'] == 'events':
            for ev_date in ev['dates']:
                if ev_date['start'] <= stop and ev_date['end'] >= start:
                    return True
        else:
            raise RuntimeError(f'Unknown data_type for event {ev}')
        return False

    def __call__(self, tfidf_scores, tag_events_scores, cluster_events_scores, slots, *args, **kwargs):
        res = []
        for tfidf_scores, tag_events_scores, cluster_events_scores, slots in zip(tfidf_scores, tag_events_scores,
                                                                                 cluster_events_scores, slots):
            tfidf_scores = self._normalize_scores(tfidf_scores)
            tag_events_scores = self._normalize_scores(tag_events_scores)
            cluster_events_scores = self._normalize_scores(cluster_events_scores)
            scores = [(k, tfidf_scores[k], tag_events_scores[k], cluster_events_scores[k])
                      for k in tfidf_scores
                      if self._filter_time_span(k, slots['time_span'])]

            scores = sorted(scores, key=lambda x: self._compute_score(x[1:]), reverse=True)

            res.append([{'title': self.data[id]['title'],
                         'local_id': id,
                         'url': self.data[id]['site_url'],
                         'tags': self.data[id]['tags'],
                         'score': self._compute_score([tfidf_score, tag_score, cluster_score]),
                         'tfidf_score': tfidf_score,
                         'tag_score': tag_score,
                         'cluster_score': cluster_score
                         } for id, tfidf_score, tag_score, cluster_score in scores[:self.n_top]])
        return res

    def _compute_score(self, scores):
        w = self.tfidf_weight + self.tag_weight + self.cluster_weight
        return (scores[0] * self.tfidf_weight + scores[1] * self.tag_weight + scores[2] * self.cluster_weight) / w

    @staticmethod
    def _normalize_scores(scores):
        w = max(max(scores.values()), 1e-06)
        for k in scores:
            scores[k] /= w
        return scores
