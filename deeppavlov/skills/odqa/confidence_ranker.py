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

from typing import List

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.component import Component

logger = get_logger(__name__)


@register("confidence_ranker")
class ConfidenceRanker(Component):
    """
    Get reader scored predictions and get top-1 prediction.
    """

    def __init__(self, reader: Component, **kwargs):
        self.reader = reader

    def __call__(self, batch_questions: List[str], batch_contexts: List[List[str]], n=1):
        batch_answers = []
        for question, contexts in zip(batch_questions, batch_contexts):
            questions = [question] * len(contexts)
            instances = list(zip(contexts, questions))
            answers = []
            for instance in instances:
                answer = self.reader([instance])
                answers += answer
            top_answer = max(answers, key=lambda item: item[2])[0]
            batch_answers.append([top_answer])
        return batch_answers
