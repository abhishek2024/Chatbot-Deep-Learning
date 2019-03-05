from pathlib import Path

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.commands.train import get_iterator_from_config, read_data_by_config
from deeppavlov.core.commands.utils import parse_config

model_config = Path(
    "/home/mks/projects/DeepPavlov/deeppavlov/configs/coreference_resolution/infer_test/gold_elmo2_fold_1.json")

config = parse_config(model_config)
model = build_model(config)
train_config = config.get("train")

data = read_data_by_config(config)
iterator = get_iterator_from_config(config, data)

# calculate outputs
expected_outputs = list(set(model.out_params))
pred_dict = dict()
for key in expected_outputs:
    pred_dict[key] = []

for x, y_true in iterator.gen_batches(1, "test", shuffle=False):
    y_predicted = list(model.compute(list(x), list(y_true), targets=expected_outputs))
    for key, pred in zip(expected_outputs, y_predicted):
        pred_dict[key].append(pred)

model.destroy()

# _, gold_clusters = get_data(config, mode)
predicted_clusters = [clusters[0] for clusters in pred_dict['predicted_clusters']]
sentences = [clusters[0] for clusters in pred_dict['sentences']]

predicted_clusters = predicted_clusters[0]
sentences = sentences[0]
# ----------------------------------------------------------------------------------------------------------------------
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

print()
