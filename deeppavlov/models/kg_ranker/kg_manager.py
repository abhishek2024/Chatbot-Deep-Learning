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


@register('kg_manager')
def KudaGoDialogueManager(Component):
    def __init__(self, cluster_policy, min_num_events, *args, **kwargs):
        self.cluster_policy = cluster_policy
        self.min_num_events = min_num_events

    def __call__(self, events, slots, utter_history):
        if len(events) < self.min_num_events:
            return events, None
        message, cluster_id = self.cluster_policy(events)
        if cluster_id is None:
            return events, None
        return message, cluster_id
