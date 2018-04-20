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

import random
import collections
import numpy as np

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register('kg_manager')
class KudaGoDialogueManager(Component):
    def __init__(self, cluster_policy, n_top, min_num_events, max_num_filled_slots,
                 *args, **kwargs):
        self.cluster_policy = cluster_policy
        self.num_top = n_top
        self.min_num_events = min_num_events
        self.max_num_filled_slots = max_num_filled_slots

    def __call__(self, events, slots, utter_history):
        messages, out_events, new_slots, cluster_ids = [], [], [], []
        for events, slots, utter_history in zip(events, slots, utter_history):
            m, ev, sl, cl_id = "", [], slots, None
            filled_slots = {s: val for s, val in slots.items() if val is not None}
            log.debug("Slots = {}".format(filled_slots))
            log.debug("Received {} events :".format(len(events)))
            for e in events[:5]:
                log.debug("score = {1:.2f}, tf_idf_score = {2:.2f}"
                          ", tag_score = {3:.2f}, cluster_score = {4:.2f}: {0}"
                          .format(e['title'], e['score'], e['tfidf_score'],
                                  e['tag_score'], e['cluster_score']))
            if not events:
                m = "Извини, нет событий, удовлетворяющих текущим условиям."\
                    " Начнем c чистого листа? Куда бы хотел сходить?"
                ev, sl, cl_id = [], {}, None
# TODO: maybe do wiser and request change of one of the slots
            elif (len(events) < self.min_num_events) or\
                    (len(filled_slots) > self.max_num_filled_slots):
                m = "Вот наиболее подходящее событие! Хотите что-то изменить?"
                ev, sl, cl_id = events[:self.num_top], slots, None
            else:
                message, cluster_id = self.cluster_policy([events], [slots])
                message, cluster_id = message[0], cluster_id[0]
                if cluster_id is None:
                    log.debug("Cluster policy didn't work: cluster_id = None")
                    m = "Вот наиболее подходящее событие! Хотите что-то изменить?"
                    ev, sl, cl_id = events[:self.num_top], slots, None
                else:
                    log.debug("Requiring cluster_id = {}".format(cluster_id))
                    m = "Вот наиболее подходящее событие. Или " + message.lower() 
                    ev, sl, cl_id = events[:self.num_top], slots, cluster_id
            messages.append(m)
            out_events.append(ev)
            new_slots.append(sl)
            cluster_ids.append(cl_id)
        return messages, out_events, new_slots, cluster_ids


@register('kg_cluster_policy')
class KudaGoClusterPolicyManager(Component):
    def __init__(self, data, tags=None, min_rate=0.01, max_rate=0.99, *args, **kwargs):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.tags_l = tags

        clusters = {cl_id: cl for cl_id, cl in data['slots'].items()
                    if cl['type'] == 'ClusterSlot'}
        if self.tags_l is None:
            self.tags_l = list(set(t for cl in clusters.values() for t in cl['tags']))
        self.questions_d = {cl_id: cl['questions'] for cl_id, cl in clusters.items()}
        # clusters: (num_clusters, num_tags)
        self.clusters_oh = {cl_id: self._onehot([cl['tags']], self.tags_l)
                            for cl_id, cl in clusters.items()}

    @staticmethod
    def _onehot(tags, all_tags):
        """
        tags: list of lists of str tags
        all_tags: list of str tags
        Returns:
            np.array (num_samples, num_tags)
        """
        num_samples, num_tags = len(tags), len(all_tags)
        onehoted = np.zeros((num_samples, num_tags))
        for i, s_tags in enumerate(tags):
            s_filtered_tags = set.intersection(set(s_tags), set(all_tags))
            for t in s_filtered_tags:
                onehoted[i, all_tags.index(t)] = 1
        return onehoted

    def __call__(self, events, slots):
        questions, cluster_ids = [], []
        for events_l, slots_d in zip(events, slots):
            event_tags_l = [e['tags'] for e in events_l if e['tags']]
            event_tags_oh = self._onehot(event_tags_l, self.tags_l)

            exclude_clusters = [cl_id for cl_id, val in slots_d.items()
                                if (cl_id not in ('time_span')) and (val is not None)]
            #if len(exclude_clusters) < 1:
            #    exclude_clusters.extend((cl_id for cl_id in slots_d.keys()
            #                             if cl_id.startswith("tag_")))
            log.debug("Excluding cluster ids: {}".format(exclude_clusters))
            clusters_oh = {cl_id: cl_oh
                           for cl_id, cl_oh in self.clusters_oh.items()
                           if cl_id not in exclude_clusters}

            bst_cluster_id, bst_rate = \
                self._best_split(event_tags_oh, clusters_oh)
            log.debug("best tag split with cluster_id = {} and rate = {}"
                      .format(bst_cluster_id, bst_rate))
            if (bst_rate < self.min_rate) or (bst_rate > self.max_rate):
                questions.append("")
                cluster_ids.append(None)
            else:
                questions.append(random.choice(self.questions_d[bst_cluster_id]))
                cluster_ids.append(bst_cluster_id)
        return questions, cluster_ids

    @classmethod
    def _best_split(cls, event_tags_oh, clusters_oh):
        """
        event_tags_oh: np.array (num_samples, num_tags)
        clusters_oh: dict with cluster ids as keys and
                     np.arrays of size (1, num_tags) as values
        Returns:
            cluster_id: str,
            divide_rate: float
        """
        cluster_ids = []
        split_rates = []
        num_events = cls._num_events_with_tags(event_tags_oh)
        for cl_id, cl_oh in clusters_oh.items():
            cluster_ids.append(cl_id)
            split_event_tags_oh = cls._split_by_tags(event_tags_oh, cl_oh)
            num_split_events = cls._num_events_with_tags(split_event_tags_oh)
            split_rates.append(num_split_events / num_events)
        best_idx = np.argmin(np.fabs(0.5 - np.array(split_rates)))
        return cluster_ids[best_idx], split_rates[best_idx]

    @staticmethod
    def _split_by_tags(event_tags_oh, tags_oh):
        """
        event_tags_oh: np.array (num_samples x num_tags)
        tags_oh: np.array (num_tags x 1) or (num_tags)
        Returns:
            np.array (num_samples x num_tags)
        """
        return np.multiply(event_tags_oh, tags_oh)

    @staticmethod
    def _num_events_with_tags(event_tags_oh):
        """
        event_tags_oh: np.array (num_samples x num_tags)
        Returns:
            int
        """
        return np.sum(np.sum(event_tags_oh, axis=1) > 0)
