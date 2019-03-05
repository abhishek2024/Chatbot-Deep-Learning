from typing import List, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register("coref_prepare")
class CorefPreprocess(Component):
    def __init__(self):
        pass

    def __call__(self, utterance_batch: List[List[str]], history_batch: List[List[List[str]]], **kwargs):
        model_batch = []
        for i, (utterance, history) in enumerate(zip(utterance_batch, history_batch)):
            model_batch.append(history.append(utterance))
        return model_batch


@register("mention_padding")
class MentionPadding(Component):
    def __init__(self):
        pass

    def __call__(self, model_batch: List[List[Tuple]], full_history: List[List[List[str]]], **kwargs):
        output_batch = []
        for (history, predicted_clusters) in zip(model_batch, full_history):
            padded_history = self.padding(history, predicted_clusters)
            output_batch.append(padded_history[-1])
        return output_batch

    @staticmethod
    def padding(sentences, predicted_clusters):
        flatten_sentences = [token for sent in sentences for token in sent]
        pos_list = []
        for sent in sentences:
            for j, token in enumerate(sent):
                if j == 0:
                    pos_list.append("sent_start")
                elif j == len(sent) - 1:
                    pos_list.append("sent_end")
                else:
                    pos_list.append("-")

        for cluster in predicted_clusters:
            cluster = sorted(cluster)
            st_mention_start = cluster[0][0]
            st_mention_end = cluster[0][1] + 1
            st_mention = flatten_sentences[st_mention_start:st_mention_end]
            for mention in cluster[1:]:
                start_mention = False
                end_mention = False
                if mention[0] == mention[1]:
                    flatten_sentences.pop(mention[0])
                    tag = pos_list.pop(mention[0])
                    if tag == "sent_start":
                        start_mention = True
                    elif tag == "sent_end":
                        end_mention = True
                else:
                    for ind in range(mention[0], mention[1], 1):
                        flatten_sentences.pop(ind)
                        tag = pos_list.pop(ind)
                        if tag == "sent_start":
                            start_mention = True
                        elif tag == "sent_end":
                            end_mention = True

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

        return new_sentences
