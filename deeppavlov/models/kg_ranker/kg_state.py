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

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('kg_state')
class KudaGoState(Component):
    def __init__(self, data, *args, **kwargs):
        self.slots = data['slots']
        self.states = {}

    def __call__(self, utterances, user_ids=None, *args, **kwargs):
        user_ids = user_ids or range(len(utterances))
        res = []
        for user_id, utterance in zip(user_ids, utterances):
            if user_id not in self.states:
                self.states[user_id] = {
                    "slots": {k: None for k in self.slots.keys()},
                    "utterances_history": [],
                    "expected_slot": None
                }
            state = self.states[user_id]
            state['utterances_history'].append(utterance)
            res.append((state['utterances_history'], state['slots'], state['expected_slot'], user_id))
        return tuple(zip(*res))


@register('kg_state_saver')
class KudaGoStateSaver(Component):
    def __init__(self, states, *args, **kwargs):
        self.states = states

    def __call__(self, user_ids, slots, expected_slot):
        for user_id, slots, expected_slot in zip(user_ids, slots, expected_slot):
            self.states[user_id]['slots'] = slots
            self.states[user_id]['expected_slot'] = expected_slot
        return user_ids
