from copy import copy
from typing import Iterable, List, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator


@register("coref_prepare")
class CorefPreprocess(Component):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, utterance_batch: List[List[str]], history_batch: List[List[List[str]]], **kwargs):
        model_batch = []
        for i, (utterance, history) in enumerate(zip(utterance_batch, history_batch)):
            history.append(utterance)
            model_batch.append(history)
        return model_batch


@register("coref_confidence")
class CorefConfidence(Component):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch: List[List[str]], **kwargs):
        return [1.0] * len(batch)


@register("mention_padding")
class MentionPadding(Component):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, model_batch: List[List[Tuple]], clusters: List[List[List[str]]], **kwargs):
        output_batch = []
        for (history, predicted_clusters) in zip(model_batch, clusters):
            padded_history = self.padding(history, predicted_clusters)
            output_batch.append(padded_history[-1])
        return output_batch

    @staticmethod
    def padding(sentences, predicted_clusters):
        flatten_sentences = [token for sent in sentences for token in sent]
        copy_flatten_sentences = copy(flatten_sentences)
        pos_list = []
        for sent in sentences:
            for j, token in enumerate(sent):
                if j == 0:
                    pos_list.append("sent_start")
                elif j == len(sent) - 1:
                    pos_list.append("sent_end")
                else:
                    pos_list.append("-")

        deviation = 0
        for cluster in predicted_clusters:
            cluster = sorted(cluster)
            st_mention_start = cluster[0][0] + deviation
            st_mention_end = cluster[0][1] + 1 + deviation
            std_len = st_mention_end - st_mention_start
            st_mention = copy_flatten_sentences[st_mention_start:st_mention_end]
            for mention in cluster[1:]:
                start_mention = False
                end_mention = False
                if mention[0] == mention[1]:
                    copy_flatten_sentences.pop(mention[0] + deviation)
                    tag = pos_list.pop(mention[0] + deviation)
                    if tag == "sent_start":
                        start_mention = True
                    elif tag == "sent_end":
                        end_mention = True
                else:
                    for ind in range(mention[0], mention[1] + 1, 1):
                        copy_flatten_sentences.pop(ind + deviation)
                        tag = pos_list.pop(ind + deviation)
                        if tag == "sent_start":
                            start_mention = True
                        elif tag == "sent_end":
                            end_mention = True

                deviation += std_len - (mention[1] - mention[0] + 1)

                for k, tok in enumerate(st_mention):
                    copy_flatten_sentences.insert(mention[0] + k + deviation, tok)
                    if start_mention and k == 0:
                        pos_list.insert(mention[0] + k + deviation, "sent_start")
                    elif end_mention and k == len(st_mention) - 1:
                        pos_list.insert(mention[0] + k + deviation, "sent_end")
                    else:
                        pos_list.insert(mention[0] + k + deviation, "-")

        assert len(pos_list) == len(copy_flatten_sentences)
        new_sentences = []
        sent = []
        for (token, tag) in zip(copy_flatten_sentences, pos_list):
            if tag == "sent_end":
                sent.append(token)
                new_sentences.append(sent)
                sent = []
            else:
                sent.append(token)

        return new_sentences


@register("mention_padding_ner")
class MentionPaddingNer(Component):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, model_batch: List[List[Tuple]], clusters_batch: List[List[List[str]]],
                 ner_batch: List[List[Tuple]],
                 **kwargs):
        output_batch = []
        for (history, predicted_clusters, ner) in zip(model_batch, clusters_batch, ner_batch):
            padded_history = self.padding(history, predicted_clusters, ner)
            output_batch.append(padded_history[-1])
        return output_batch

    @staticmethod
    def padding(sentences, predicted_clusters, ner_tokens):
        flatten_sentences = [token for sent in sentences for token in sent]
        flatten_ner_tokens = [token for sent in ner_tokens for token in sent]

        old_clusters = {}
        for i, cluster in enumerate(predicted_clusters):
            tmp_list = []
            for mention in cluster:
                tmp_list.append(list(flatten_sentences[mention[0]:mention[-1] + 1]))
            old_clusters[i] = tmp_list

        pos_list = []
        for sent in sentences:
            for j, token in enumerate(sent):
                if j == 0:
                    pos_list.append("sent_start")
                elif j == len(sent) - 1:
                    pos_list.append("sent_end")
                else:
                    pos_list.append("-")

        cluster_names = dict()
        for i, cluster in enumerate(predicted_clusters):
            for mention in cluster:
                if mention[0] == mention[1]:
                    if flatten_ner_tokens[mention[0]] == 'B-PER':
                        cluster_names[i] = flatten_sentences[mention[0]:mention[0] + 1]
                        break
                else:
                    name_end_ = mention[0]  # -1
                    for ind in range(mention[0], mention[1] + 1, 1):
                        if flatten_ner_tokens[ind] == 'B-PER' or flatten_ner_tokens[ind] == 'I-PER':
                            name_end_ = ind
                    if name_end_ != mention[0] - 1:
                        cluster_names[i] = flatten_sentences[mention[0]:name_end_ + 1]
                        break

        deviation = 0
        for i, cluster in enumerate(predicted_clusters):
            if i in cluster_names.keys():
                st_mention = cluster_names[i]
                for mention in cluster:
                    start_mention = False
                    end_mention = False
                    if mention[0] == mention[1]:
                        flatten_sentences.pop(mention[0] + deviation)
                        tag = pos_list.pop(mention[0] + deviation)
                        if tag == "sent_start":
                            start_mention = True
                        elif tag == "sent_end":
                            end_mention = True
                    else:
                        for ind in range(mention[0], mention[1] + 1, 1):
                            flatten_sentences.pop(mention[0] + deviation)
                            tag = pos_list.pop(mention[0] + deviation)
                            if tag == "sent_start":
                                start_mention = True
                            elif tag == "sent_end":
                                end_mention = True

                    deviation += len(st_mention) - (mention[1] - mention[0] + 1)
                    for k, tok in enumerate(st_mention):
                        flatten_sentences.insert(mention[0] + k, tok)
                        if start_mention and k == 0:
                            pos_list.insert(mention[0] + k, "sent_start")
                        elif end_mention and k == len(st_mention) - 1:
                            pos_list.insert(mention[0] + k, "sent_end")
                        else:
                            pos_list.insert(mention[0] + k, "-")

        assert len(pos_list) == len(flatten_sentences)
        new_sentences = []
        sent = []
        for (token, tag) in zip(flatten_sentences, pos_list):
            if tag == "sent_end":
                sent.append(token)
                new_sentences.append(sent)
                sent = []
            else:
                sent.append(token)

        # from pathlib import Path
        # logs = Path("/home/mks/Downloads/")
        # if len(cluster_names) != 0:
        #     with logs.joinpath("coref_ner_logs.txt").open("a", encoding='utf8') as log:
        #         print("### START ###", file=log)
        #         for sent in sentences:
        #             print(sent, file=log)
        #         print("\n", file=log)
        #         print(cluster_names, file=log)
        #         print("\n", file=log)
        #         print(old_clusters, file=log)
        #         print("\n", file=log)
        #         for new_sent in new_sentences:
        #             print(new_sent, file=log)
        #         print("### END ###", file=log)
        #         print("\n", file=log)

        # from pathlib import Path
        # logs = Path("/home/mks/Downloads/")
        # if len(cluster_names) != 0:
        #     with logs.joinpath("coref_count_logs.txt").open("a", encoding='utf8') as log:
        #         print(cluster_names, file=log)

        return new_sentences


class PerItemWrapper(Component):

    def __init__(self, component: Estimator, **kwargs):
        self.component = component
        self.return_out = kwargs.get('return_out')

    def __call__(self, batch):
        out = []
        if isinstance(batch[0], Iterable) and not isinstance(batch[0], str):
            for item in batch:
                if item:
                    out.append(self.component(item))
                else:
                    out.append([])
        else:
            out = self.component(batch)
        # log.info(f"in = {batch}, out = {out}")
        if self.return_out:
            return out
        else:
            return tuple(zip(*out))

    def fit(self, data):
        self.component.fit([item for batch in data for item in batch])

    def save(self, *args, **kwargs):
        self.component.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        self.component.load(*args, **kwargs)