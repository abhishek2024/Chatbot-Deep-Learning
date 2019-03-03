import shutil
from pathlib import Path
from typing import Union

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.commands.train import get_iterator_from_config, read_data_by_config
from deeppavlov.core.commands.utils import parse_config
from deeppavlov.models.coreference_resolution.compute_metrics.compute_metrics import compute_ancor_metrics


def prepare_folds(file_names, root, chain_files):
    # prepare data and folds
    if root.exists():
        shutil.rmtree(root)
    root.joinpath("pred").mkdir(parents=True)
    root.joinpath("gold").mkdir(parents=True)

    for doc_key_ in file_names:
        shutil.copy(str(chain_files.joinpath(doc_key_[2:-2] + '.txt')), str(root.joinpath("gold")))
    return root.joinpath("pred"), root.joinpath("gold")


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
        doc_map[x[0][2][2:-2]]['clusters'], _, _ = list(model.compute(list(x), list(y_true)))
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
            for i, tupl in enumerate(val):
                z = [str(c) for c in tupl]
                pred_.write(" ".join([str(i + 1), *z]) + "\n")


def models_metrics(configs, res_fold, morph_path, chains_path, mode='test'):
    gold_path = res_fold.joinpath('gold')
    new_path = res_fold.joinpath('pred')

    for i, config in enumerate(configs):
        if gold_path.exists():
            shutil.rmtree(gold_path)
        if new_path.exists():
            shutil.rmtree(new_path)

        gold_path.mkdir()
        new_path.mkdir()

        write_predictions(config, res_fold, chains_path, morph_path, mode)
        print(f"Model - {i}")
        compute_ancor_metrics(str(gold_path), str(new_path), str(res_fold))
        shutil.rmtree(new_path)
        shutil.rmtree(gold_path)


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
            for i, tup_ in enumerate(val):
                z = [str(c) for c in tup_]
                pred_.write(" ".join([str(i + 1), *z]) + "\n")


def compute_competition_metrics(some_prediction, root, chains_path, morph_paths):
    test_files = [key for key in some_prediction.keys()]
    pred_path, key_path = prepare_folds(test_files, root, chains_path)

    transform_model_predictions_in_ancor_format(some_prediction, pred_path, morph_paths)
    compute_ancor_metrics(key_path, pred_path, root)
