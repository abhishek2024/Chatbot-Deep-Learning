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


@register("kg_clusters2events_scorer")
class KGClusters2EventsScorer(Component):
    def __init__(self, data, *args, **kwargs):
        self.data = data['places_events']
        self.slots = data['slots']

    def __call__(self, batch, *args, **kwargs):
        batch_scores = []
        for slot_history in batch:
            scores = {}
            for event in self.data:
                scores[event['local_id']] = 0
                event_tags = set(event['tags'])
                for slot in self.slots:
                    if self.slots[slot]['type'] == 'ClusterSlot' and slot_history.get(slot):
                        slot_tags = set(self.slots[slot]['tags'])
                        if slot_history[slot] == 'yes':
                            scores[event['local_id']] += len(slot_tags & event_tags)
                        elif slot_history[slot] == 'no':
                            scores[event['local_id']] -= len(slot_tags & event_tags)
            batch_scores.append(scores)
        return batch_scores
