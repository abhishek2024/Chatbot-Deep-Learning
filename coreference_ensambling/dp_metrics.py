from copy import deepcopy

from coreference_ensambling.ensambling import get_data
from coreference_ensambling.new_ensembling import get_iterator, get_model, get_predicted_antecedents, \
    get_predicted_clusters
from deeppavlov.core.commands.train import _parse_metrics
from deeppavlov.metrics.coref_metrics import coref_metrics


def compute_dp_metric_for_ensemble(ensemble_clusters, ensemble_mentions_to_predict, data_config, mode='test'):
    predicted_clusters = [val for key, val in ensemble_clusters.items()]
    mentions_to_predict = [val for key, val in ensemble_mentions_to_predict.items()]

    _, gold_clusters = get_data(data_config, mode)

    gold_clusters = [clusters[0] for clusters in gold_clusters]

    print(f"DP - F1: {coref_metrics(predicted_clusters, mentions_to_predict, gold_clusters)}")
    # metrics = [(m.name, m.fn(*[outputs[i] for i in m.inputs])) for m in metrics_functions]


# TODO fix (very low score, something wrong)
def get_model_prediction_from_ensamble_call(config, mode='test', batch_size=1):
    model, chainer, config = get_model(config)
    iterator = get_iterator(config)
    if isinstance(config['chainer']['out'], str):
        config['chainer']['out'] = [config['chainer']['out']]

    expected_outputs = list(set(chainer.out_params))
    pred_dict = dict()
    for key in expected_outputs:
        pred_dict[key] = []

    doc_clusters, doc_mentions_to_predicted = [], []
    for x, y_true in iterator.gen_batches(batch_size, mode, shuffle=False):
        y_predicted = list(chainer.compute(list(x), list(y_true), targets=expected_outputs))
        for key, pred in zip(expected_outputs, y_predicted):
            pred_dict[key].append(pred)

        ensemble_predicted_antecedents = get_predicted_antecedents(pred_dict['top_antecedents'][0],
                                                                   pred_dict['top_antecedent_scores'][0])
        predicted_clusters, mention_to_predicted = get_predicted_clusters(pred_dict['top_span_starts'][0],
                                                                          pred_dict['top_span_ends'][0],
                                                                          ensemble_predicted_antecedents)
        doc_clusters.append(predicted_clusters)
        doc_mentions_to_predicted.append(mention_to_predicted)

    chainer.destroy()
    return doc_clusters, doc_mentions_to_predicted


def get_model_prediction_from_simple_call(config, mode='test', batch_size=1):
    model, chainer, config = get_model(config)
    iterator = get_iterator(config)
    in_y = config['chainer'].get('in_y', ['y'])
    if isinstance(in_y, str):
        in_y = [in_y]
    if isinstance(config['chainer']['out'], str):
        config['chainer']['out'] = [config['chainer']['out']]
    metrics_functions = _parse_metrics(config['train']['metrics'], in_y, config['chainer']['out'])

    expected_outputs = list(set().union(chainer.out_params, *[m.inputs for m in metrics_functions]))
    pred_dict = dict()
    for key in expected_outputs:
        pred_dict[key] = []

    for x, y_true in iterator.gen_batches(batch_size, mode, shuffle=False):
        y_predicted = list(chainer.compute(list(x), list(y_true), targets=expected_outputs))
        for key, pred in zip(expected_outputs, y_predicted):
            pred_dict[key].append(pred)

    chainer.destroy()

    return pred_dict


def compute_dp_metric_for_models(configs, mode='test'):
    for k, config in enumerate(configs):
        prediction_dict = get_model_prediction_from_simple_call(deepcopy(config), mode)
        mention_to_predicted = prediction_dict['mention_to_predicted']
        predicted_clusters = prediction_dict['predicted_clusters']
        gold_clusters = prediction_dict['clusters']

        # _, gold_clusters = get_data(config, mode)
        gold_clusters = [clusters[0] for clusters in gold_clusters]
        predicted_clusters = [clusters[0] for clusters in predicted_clusters]
        mention_to_predicted = [clusters[0] for clusters in mention_to_predicted]
        print(f"DP - F1 for model {k}: {coref_metrics(predicted_clusters, mention_to_predicted, gold_clusters)}")
