import shutil
from pathlib import Path
from typing import Union

from copy import deepcopy

import numpy as np

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.commands.train import get_iterator_from_config, read_data_by_config
from deeppavlov.core.commands.utils import parse_config
from deeppavlov.models.coreference_resolution.compute_metrics.compute_metrics import compute_ancor_metrics


def write_predictions(config_path: Union[str, Path], results_path: Union[str, Path], chains_folder: Union[str, Path],
                      morph_folder: Union[str, Path], data_type: str = "test") -> None:
    key_file_path = results_path.joinpath('gold')
    pred_file_path = results_path.joinpath('pred')

    config = parse_config(config_path)
    data = read_data_by_config(config)
    generator = get_iterator_from_config(config, data)
    model = build_model(config_path)

    doc_map = {}
    for x, y_true in generator.gen_batches(1, data_type, shuffle=False):
        doc_map[x[0][2][2:-2]] = dict()
        doc_map[x[0][2][2:-2]]['clusters'], _ = list(model.compute(list(x), list(y_true)))
        doc_map[x[0][2][2:-2]]['clusters'] = [tuple(set(cluster)) for cluster in doc_map[x[0][2][2:-2]]['clusters'][0]]

    model.destroy()

    ancor_format = {}
    for doc, val in doc_map.items():
        file_name = doc + '.txt'
        ancor_format[file_name] = []
        shutil.copy(str(chains_folder.joinpath(file_name)), str(key_file_path))

        with morph_folder.joinpath(file_name).open("r") as file_:
            doc_lines = [line.strip().split('\t') for line in file_ if line.strip()]

        for i, cluster in enumerate(val['clusters']):
            for mention in cluster:
                ancor_format[file_name].append((int(doc_lines[mention[0]][1]),
                                                (int(doc_lines[mention[1]][1]) + int(doc_lines[mention[1]][2]) - int(
                                                    doc_lines[mention[0]][1])),
                                                i + 1))

        ancor_format[file_name] = sorted(ancor_format[file_name], key=lambda tup: tup[0])

    for key, val in ancor_format.items():
        with pred_file_path.joinpath(key).open("w", encoding='utf8') as pred_:
            for i, tup in enumerate(val):
                z = [str(c) for c in tup]
                pred_.write(" ".join([str(i + 1), *z]) + "\n")


def get_data(config, mode, parse_conf=True):
    if parse_conf:
        config = parse_config(config)
    data = read_data_by_config(config)
    generator = get_iterator_from_config(config, data)
    data_for_model = []
    gold_predictions = []
    for x, y_true in generator.gen_batches(1, mode, shuffle=False):
        data_for_model.append(x)
        gold_predictions.append(y_true)
    return data_for_model, gold_predictions


def get_model(model_config):
    config = parse_config(model_config)
    chainer = build_model(config)
    return chainer.get_main_component(), chainer, config


def get_predicted_antecedents(antecedents, antecedent_scores):
    predicted_antecedents = []
    for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
        if index < 0:
            predicted_antecedents.append(-1)
        else:
            predicted_antecedents.append(antecedents[i, index])
    return predicted_antecedents


def get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
        if predicted_index < 0:
            continue
        assert i > predicted_index
        predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
        if predicted_antecedent in mention_to_predicted:
            predicted_cluster = mention_to_predicted[predicted_antecedent]
        else:
            predicted_cluster = len(predicted_clusters)
            predicted_clusters.append([predicted_antecedent])
            mention_to_predicted[predicted_antecedent] = predicted_cluster

        mention = (int(top_span_starts[i]), int(top_span_ends[i]))
        predicted_clusters[predicted_cluster].append(mention)
        mention_to_predicted[mention] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

    return predicted_clusters, mention_to_predicted


def mean_prediction(docs_dict, num_models):
    # parse predictions, spans, antecedents and antecedents_scores
    docs_predictions = {}
    for doc_name, model_predictions in docs_dict.items():
        doc_structure = {}
        for model_pred in model_predictions:
            padding = (model_pred['top_span_starts'][0], model_pred['top_span_ends'][0])
            for i, span in enumerate(zip(*(model_pred['top_span_starts'], model_pred['top_span_ends']))):
                # cleaning padding
                if span == padding and i != 0:
                    continue
                if span not in doc_structure:
                    doc_structure[span] = {}
                for k, span_ind in enumerate(model_pred['top_antecedents'][i]):
                    span_id = (model_pred['top_span_starts'][span_ind], model_pred['top_span_ends'][span_ind])
                    # cleaning padding
                    if span_id == padding and k != 0:
                        continue
                    if span_id not in doc_structure[span]:
                        doc_structure[span][span_id] = []
                    doc_structure[span][span_id].append(
                        model_pred['top_antecedent_scores'][i][k + 1])  # (k + 1) CHECK IT!!!!!!!!!!!!!!!!!!!!
        docs_predictions[doc_name] = deepcopy(doc_structure)

    # transform docs_predictions in model format
    ensemble_predictions = {}
    for doc_name, doc_structure in docs_predictions.items():
        ensemble_top_spans = list(doc_structure.keys())
        ensemble_top_span_starts, ensemble_top_span_ends, ensemble_top_antecedents, ensemble_top_antecedent_scores = [], [], [], []
        for span, antecedents in doc_structure.items():
            ensemble_top_span_starts.append(span[0])
            ensemble_top_span_ends.append(span[1])
            span_antecedents = []
            span_antecedents_scores = [0.]  # 0.(k + 1) CHECK IT!!!!!!!!!!!!!!!!!!!!
            for antecedent_span_id, antecedent_models_scores in antecedents.items():
                span_antecedents.append(ensemble_top_spans.index(antecedent_span_id))
                span_antecedents_scores.append(np.sum(antecedent_models_scores) / num_models)  # MEANING
            ensemble_top_antecedents.append(span_antecedents)
            ensemble_top_antecedent_scores.append(span_antecedents_scores)

        # create [predicted_clusters, mention_to_predicted] from new ensemble res
        # add paddings
        n_paddings = len(ensemble_top_span_starts) - min([len(el) for el in ensemble_top_antecedents])
        ensemble_top_span_starts += [ensemble_top_span_starts[0]] * n_paddings
        ensemble_top_span_ends += [ensemble_top_span_ends[0]] * n_paddings
        padded_len = len(ensemble_top_span_starts)

        for i in range(len(ensemble_top_antecedents)):
            ensemble_top_antecedents[i] = ensemble_top_antecedents[i] + list(range(len(ensemble_top_antecedents[i]) + 1, padded_len))

        for i in range(len(ensemble_top_antecedent_scores)):
            ensemble_top_antecedent_scores[i] = ensemble_top_antecedent_scores[i] + [float('-inf')] * (padded_len - len(ensemble_top_antecedent_scores[i]))

        ensemble_top_antecedents = np.array(ensemble_top_antecedents)
        ensemble_top_antecedent_scores = np.array(ensemble_top_antecedent_scores)
        if len(ensemble_top_antecedents.shape) == 1 and len(ensemble_top_antecedent_scores.shape) == 1:
            ensemble_top_antecedent_scores = np.reshape(ensemble_top_antecedent_scores,
                                                        (ensemble_top_antecedent_scores.shape[0], 1))
            ensemble_top_antecedents = np.reshape(ensemble_top_antecedents, (ensemble_top_antecedents.shape[0], 1))

        ensemble_predicted_antecedents = get_predicted_antecedents(ensemble_top_antecedents,
                                                                   ensemble_top_antecedent_scores)
        predicted_clusters, mention_to_predicted = get_predicted_clusters(ensemble_top_span_starts,
                                                                          ensemble_top_span_ends,
                                                                          ensemble_predicted_antecedents)

        ensemble_predictions[doc_name] = predicted_clusters
    return ensemble_predictions


def get_ensemble_prediction(models_configs, data):
    docs = {}
    doc_names = []
    model_predictions = {}
    num_models = len(models_configs)
    # get predictions
    for model_ind, model_config in enumerate(models_configs):
        model, chainer, config = get_model(model_config)
        model_predictions[model_ind] = {"config": config, "docs": {}}
        for batch in data:
            sentences, speakers, doc_key, mentions_positions = batch[0]
            if doc_key not in doc_names:
                doc_names.append(doc_key)
            top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = \
                model.predict_for_ensemble(sentences, speakers, doc_key, mentions_positions)
            pred_dict = dict(top_span_starts=top_span_starts,  # k
                             top_span_ends=top_span_ends,  # k
                             top_antecedents=top_antecedents,  # [k,c]
                             top_antecedent_scores=top_antecedent_scores)  # [k,c+1]
            model_predictions[model_ind]["docs"][doc_key] = pred_dict
        chainer.destroy()

    # reformat model_predictions to docs_predictions
    for model_ind, model_dict in model_predictions.items():
        for doc_key, doc_prediction in model_predictions[model_ind]["docs"].items():
            if doc_key not in docs:
                docs[doc_key] = []
            docs[doc_key].append(doc_prediction)

    # calculate ensemble predictions
    ensemble_predicted_clusters = mean_prediction(docs, num_models)
    return ensemble_predicted_clusters  # {"doc_name": clusters}


def transform_model_predictions_in_ancor_format(docs_dict, results_path, morph_fold, remove_mini_clusters=False):
    doc_map = {}
    for name_doc, clusters in docs_dict.items():
        doc_map[name_doc[2:-2]] = clusters

    # delete clusters with len == 1
    if remove_mini_clusters:
        for name_doc, clusters in doc_map.items():
            for i, cluster in enumerate(clusters):
                if len(cluster) == 1:
                    clusters.remove(cluster)

    ancor_format = {}
    for doc, val in doc_map.items():
        file_name = doc + '.txt'
        ancor_format[file_name] = []

        with morph_fold.joinpath(file_name).open("r") as file_:
            doc_lines = [line.strip().split('\t') for line in file_ if line.strip()]

        for i, cluster in enumerate(val):
            for mention in cluster:
                ancor_format[file_name].append((int(doc_lines[mention[0]][1]),
                                                (int(doc_lines[mention[1]][1]) + int(doc_lines[mention[1]][2]) - int(
                                                    doc_lines[mention[0]][1])),
                                                i + 1))

        ancor_format[file_name] = sorted(ancor_format[file_name], key=lambda tup: tup[0])

    for key, val in ancor_format.items():
        with results_path.joinpath(key).open("w", encoding='utf8') as pred_:
            for i, tup in enumerate(val):
                z = [str(c) for c in tup]
                pred_.write(" ".join([str(i + 1), *z]) + "\n")


def models_metrics(configs, res_fold, morph_path, chains_path, mode='test'):
    gold_path = res_fold.joinpath('gold')
    new_path = res_fold.joinpath('pred')

    for config in configs:
        if gold_path.exists():
            shutil.rmtree(gold_path)
        if new_path.exists():
            shutil.rmtree(new_path)

        gold_path.mkdir()
        new_path.mkdir()

        write_predictions(config, res_fold, chains_path, morph_path, mode)
        print(f"Model - {config.name}")
        compute_ancor_metrics(str(gold_path), str(new_path), str(res_fold))
        shutil.rmtree(new_path)
        shutil.rmtree(gold_path)


# ensemble
ensemble_root = Path("/home/mks/projects/DeepPavlov/deeppavlov/configs/coreference_resolution/test_model/")
ancor_data = Path("/home/mks/Work/Datasets/AnCor/ForTraining+morph/")
chains_files = ancor_data.joinpath("Chains")
morph_files = ancor_data.joinpath("morph")
root = Path("/home/mks/Downloads/tmp")

pred_path = root.joinpath("pred")
key_path = root.joinpath("gold")

ensemble_configs = [path for path in ensemble_root.glob("*.json")]
assert len(ensemble_configs) >= 1

# ensemble prediction
data, gold_preds = get_data(ensemble_configs[0], "test")
ensemble_prediction = get_ensemble_prediction([ensemble_configs[0]], data)

if pred_path.exists():
    shutil.rmtree(pred_path)
if key_path.exists():
    shutil.rmtree(key_path)

key_path.mkdir()
pred_path.mkdir()

for doc_key_, _ in ensemble_prediction.items():
    shutil.copy(str(chains_files.joinpath(doc_key_[2:-2] + '.txt')), str(key_path))

transform_model_predictions_in_ancor_format(ensemble_prediction, pred_path, morph_files)
compute_ancor_metrics(key_path, pred_path, root)

# Alone Models
models_metrics([ensemble_configs[0]], root, morph_files, chains_files)
