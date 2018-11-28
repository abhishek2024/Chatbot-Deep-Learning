# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
import subprocess
import numpy as np

from typing import List
from pathlib import Path
from shutil import rmtree
from os.path import join, exists
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.metrics_registry import register_metric

log = get_logger(__name__)

metrics = ("muc", "bcub", "ceafe")
BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")
COREF_RESULTS_REGEX = re.compile(r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ "
                                 r"/ [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)


def official_conll_eval(gold_path, predicted_path, metric):
    scorer_path = join(str(Path(__file__).resolve().parent), "./scorer/v8.01/scorer.pl")
    cmd = [scorer_path, metric, gold_path, predicted_path, "none"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    if stderr is not None:
        print(stderr)

    coref_results_match = re.match(COREF_RESULTS_REGEX, stdout)
    recall = float(coref_results_match.group(1))
    precision = float(coref_results_match.group(2))
    f1 = float(coref_results_match.group(3))
    return {"r": recall, "p": precision, "f": f1}


@register_metric('coref_f1')
def coref_f1(gold_strings: List[str], predicted_strings: List[str]):
    gold_path = "./tmp/gold/"
    predicted_path = "./tmp/pred/"

    if exists("./tmp"):
        rmtree("./tmp")

    os.makedirs(gold_path)
    os.makedirs(predicted_path)

    for i, gold_str in enumerate(gold_strings):
        with open(join(gold_path, f"{i+1}.gold_conll"), "w") as gf:
            gf.write(gold_str)

    for i, pred_str in enumerate(predicted_strings):
        with open(join(predicted_path, f"{i+1}.gold_conll"), "w") as pf:
            pf.write(pred_str)

    results = []
    for file_name in os.listdir(gold_path):
        gold_file = join(gold_path, file_name)
        pred_file = join(predicted_path, file_name)

        tmp_res = {}
        for met in metrics:
            tmp_res[met] = official_conll_eval(gold_file, pred_file, met)

        conll_f1 = 0
        conll_p = 0
        conll_r = 0
        for met in metrics:
            conll_f1 += tmp_res[met]['f']
            conll_p += tmp_res[met]['p']
            conll_r += tmp_res[met]['r']

        tmp_res['Conll F1'] = {}
        tmp_res['Conll F1']['r'] = conll_r / len(metrics)
        tmp_res['Conll F1']['p'] = conll_p / len(metrics)
        tmp_res['Conll F1']['f'] = conll_f1 / len(metrics)
        results.append(tmp_res)

    men_res = {}
    for met in results[0].keys():
        men_res[met] = []
        for file_res in results:
            men_res[met].append(file_res[met]['f'])

    mean_res = {}
    for met in men_res.keys():
        mean_res[met] = np.mean(np.array(men_res[met]))

    rmtree("./tmp")

    return mean_res['Conll F1']
