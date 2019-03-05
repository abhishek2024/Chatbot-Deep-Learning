import json
from copy import deepcopy
from pathlib import Path
from typing import Union

from coreference_ensambling.ensembling import get_iterator, get_model
from deeppavlov.core.commands.train import _parse_metrics


def get_model_docs_prediction(config, mode='test', batch_size=1):
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
        pred_dict[key] = {}

    for x, y_true in iterator.gen_batches(batch_size, mode, shuffle=False):
        y_predicted = list(chainer.compute(list(x), list(y_true), targets=expected_outputs))
        for key, pred in zip(expected_outputs, y_predicted):
            pred_dict[key][x[0][2][2:-2]] = pred

    chainer.destroy()

    return pred_dict


def write_predictions(predicted_clusters,
                      results_path: Union[str, Path],
                      morph_folder: Union[str, Path]):
    # predicted_clusters = {key: clusters[0] for key, clusters in predicted_clusters.items()}
    # predicted_clusters = {key: [set(cluster) for cluster in clusters[0]] for key, clusters in
    #                       predicted_clusters.items()}
    predicted_clusters = {key[2:-2]: clusters for key, clusters in predicted_clusters.items()}

    ancor_format = {}
    for doc, val in predicted_clusters.items():
        file_name = doc + '.txt'
        ancor_format[file_name] = []
        try:
            with morph_folder.joinpath(file_name).open("r") as file_:
                doc_lines = [line.strip().split('\t') for line in file_ if line.strip()]

            for i, cluster in enumerate(val):
                for mention in cluster:
                    ancor_format[file_name].append((int(doc_lines[mention[0]][1]),
                                                    (int(doc_lines[mention[1]][1]) + int(
                                                        doc_lines[mention[1]][2]) - int(
                                                        doc_lines[mention[0]][1])),
                                                    i + 1))

            ancor_format[file_name] = sorted(ancor_format[file_name], key=lambda tup: tup[0])
        except FileNotFoundError:
            print(morph_folder.joinpath(file_name))
            continue

    for key, val in ancor_format.items():
        try:
            with results_path.joinpath(key).open("w", encoding='utf8') as pred_:
                for i, tupl in enumerate(val):
                    z = [str(c) for c in tupl]
                    pred_.write(" ".join([str(i + 1), *z]) + "\n")
        except FileNotFoundError:
            print(key)
            continue


def prepare_ensemble(model_config, test_path, test_lm_path, ensemble=False, kfold=10, cv_tmp=False):
    # change test_part into config
    with model_config.open('r') as conf:
        config = json.load(conf)

    config['dataset_reader']['test'] = test_path
    config['dataset_iterator'] = {"class_name": "coreference_iterator", "train_on_gold": True, "seed": 42}
    config['chainer']['pipe'][-1]['lm_path'] = test_lm_path

    if ensemble:
        configs = []
        for i in range(kfold):
            config_ = deepcopy(config)
            old_load_path = config_['chainer']['pipe'][1]['load_path'].split('/')
            parent = old_load_path[:-1]
            if cv_tmp:
                vocab_name = config_['chainer']['pipe'][1]['load_path'].split('/')[-1]
                model_name = config_['chainer']['pipe'][-1]['load_path'].split('/')[-1]
                config_['chainer']['pipe'][1]['load_path'] = '/'.join(parent + ["cv_tmp", f"fold_{i + 1}", vocab_name])
                config_['chainer']['pipe'][-1]['load_path'] = '/'.join(parent + ["cv_tmp", f"fold_{i + 1}", model_name])
            else:
                vocab_name = config_['chainer']['pipe'][1]['load_path'].split('/')[-1]
                model_name = config_['chainer']['pipe'][-1]['load_path'].split('/')[-1]
                config_['chainer']['pipe'][1]['load_path'] = '/'.join(parent + [f"fold_{i + 1}", vocab_name])
                config_['chainer']['pipe'][-1]['load_path'] = '/'.join(parent + [f"fold_{i + 1}", model_name])

            # ensemble configs
            config_['chainer']['pipe'][-1]['ensemble_prediction'] = True
            config_['chainer']['pipe'][-1]['out'] = ["top_span_starts", "top_span_ends", "top_antecedents",
                                                     "top_antecedent_scores"]

            config_['chainer']['out'] = ["top_span_starts", "top_span_ends", "top_antecedents", "top_antecedent_scores"]

            configs.append(config_)
        return configs
    else:
        return config
