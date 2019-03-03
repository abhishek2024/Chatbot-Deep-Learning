from copy import deepcopy

import numpy as np

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.commands.train import get_iterator_from_config, read_data_by_config, _parse_metrics
from deeppavlov.core.commands.utils import parse_config


def get_iterator(config, parse_conf=True):
    if parse_conf:
        config = parse_config(config)
    data = read_data_by_config(config)
    return get_iterator_from_config(config, data)


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
        # assert i > predicted_index
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


def mean_prediction(docs_dict):
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
                        model_pred['top_antecedent_scores'][i][k + 1])
        docs_predictions[doc_name] = deepcopy(doc_structure)

    # transform docs_predictions in model format
    ensemble_predictions = {}
    mentions_dict = {}
    for doc_name, doc_structure in docs_predictions.items():
        ensemble_top_spans = list(doc_structure.keys())
        ensemble_top_span_starts, ensemble_top_span_ends = [], []
        ensemble_top_antecedents, ensemble_top_antecedent_scores = [], []
        for span, antecedents in doc_structure.items():
            ensemble_top_span_starts.append(span[0])
            ensemble_top_span_ends.append(span[1])
            span_antecedents = []
            span_antecedents_scores = [0.]
            for antecedent_span_id, antecedent_models_scores in antecedents.items():
                span_antecedents.append(ensemble_top_spans.index(antecedent_span_id))
                # span_antecedents_scores.append(np.sum(antecedent_models_scores) / num_models)  # MEANING
                span_antecedents_scores.append(np.median(antecedent_models_scores))  # median from models preds
            ensemble_top_antecedents.append(span_antecedents)
            ensemble_top_antecedent_scores.append(span_antecedents_scores)

        # create [predicted_clusters, mention_to_predicted] from new ensemble res
        # add paddings
        n_paddings = len(ensemble_top_span_starts) - min([len(el) for el in ensemble_top_antecedents])
        ensemble_top_span_starts += [ensemble_top_span_starts[0]] * n_paddings
        ensemble_top_span_ends += [ensemble_top_span_ends[0]] * n_paddings
        padded_len = len(ensemble_top_span_starts)

        for i in range(len(ensemble_top_antecedents)):
            ensemble_top_antecedents[i] = ensemble_top_antecedents[i] + list(
                range(len(ensemble_top_antecedents[i]) + 1, padded_len))

        for i in range(len(ensemble_top_antecedent_scores)):
            ensemble_top_antecedent_scores[i] = ensemble_top_antecedent_scores[i] + [float('-inf')] * (
                    padded_len - len(ensemble_top_antecedent_scores[i]))

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
        mentions_dict[doc_name] = mention_to_predicted
    return ensemble_predictions, mentions_dict


def get_ensemble_prediction(models_configs, mode, batch_size=1):
    docs = {}
    model_predictions = {}
    # get predictions
    for model_ind, model_config in enumerate(models_configs):
        model, chainer, config = get_model(model_config)
        iterator = get_iterator(model_config)
        model_predictions[model_ind] = {"config": config, "docs": {}}

        in_y = config['chainer'].get('in_y', ['y'])
        if isinstance(in_y, str):
            in_y = [in_y]
        if isinstance(config['chainer']['out'], str):
            config['chainer']['out'] = [config['chainer']['out']]

        # metrics_functions = _parse_metrics(config['train'].get('metrics'), in_y, config['chainer']['out'])
        # expected_outputs = list(set().union(chainer.out_params, *[m.inputs for m in metrics_functions]))
        expected_outputs = list(set(chainer.out_params))
        for x, y_true in iterator.gen_batches(batch_size, mode, shuffle=False):
            y_predicted = list(chainer.compute(list(x), list(y_true), targets=expected_outputs))
            pred_dict = dict()
            for key, pred in zip(expected_outputs, y_predicted):
                pred_dict[key] = pred

            model_predictions[model_ind]["docs"][x[0][2]] = pred_dict

        chainer.destroy()

    # reformat model_predictions to docs_predictions
    for model_ind, model_dict in model_predictions.items():
        for doc_key, doc_prediction in model_predictions[model_ind]["docs"].items():
            if doc_key not in docs:
                docs[doc_key] = []
            docs[doc_key].append(doc_prediction)

    # calculate ensemble predictions
    ensemble_predicted_clusters, mention_to_predicted = mean_prediction(docs)
    return ensemble_predicted_clusters, mention_to_predicted  # {"doc_name": clusters}
