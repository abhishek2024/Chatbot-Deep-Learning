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


@register('tags_score_fuser')
class Mask(Component):
    def __init__(self, tags, merge_b_i=True, *args, **kwargs):
        self.tags = list(tags)
        self.merge_b_i = merge_b_i

    def __call__(self, probs_batch, **kwargs):
        """Takes batch of tokens and returns the masks of corresponding length"""
        batch = [{}] * len(probs_batch)
        for k in range(len(probs_batch)):
            for n, tag in enumerate(self.tags):
                if self.merge_b_i:
                    if tag != 'O':
                        tag = tag[2:]
                    if tag in batch[k]:
                        batch[k][tag] += probs_batch[k, :, n]
                    else:
                        batch[k][tag] = probs_batch[k, :, n]
                else:
                    batch[k][tag] = probs_batch[k, :, n]

        return batch
