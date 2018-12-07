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
from typing import List, Any
from itertools import chain

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register


@register("query_paragraph_tuplifier")
class QueryParagraphTuplifier(Component):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch_queries: List[str], batch_contexts: List[List[str]],
                 batch_doc_ids: List[List[Any]], *args, **kwargs):
        tuples = []

        if len(batch_contexts) == 1:
            assert len(batch_doc_ids) == 1
            batch_contexts = [batch_contexts[0]] * len(batch_queries)
            batch_doc_ids = [batch_doc_ids[0]] * len(batch_queries)
        for query, contexts, ids in zip(batch_queries, batch_contexts, batch_doc_ids):
            context_lengths = [len(c) for c in contexts]
            flat_ids = [[ids[i]] * i for i in context_lengths]
            flat_ids = list(chain.from_iterable(flat_ids))
            contexts = list(chain.from_iterable(contexts))
            tuples.append((query, contexts, flat_ids))
        return tuples
