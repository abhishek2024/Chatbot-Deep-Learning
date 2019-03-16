# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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

import json
from collections import OrderedDict, namedtuple
from typing import List, Tuple, Union, Iterable

import numpy as np

from deeppavlov.core.common.metrics_registry import get_metric_by_name

Metric = namedtuple('Metric', ['name', 'fn', 'inputs'])


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def parse_metrics(metrics: Iterable[Union[str, dict]], in_y: List[str], out_vars: List[str]) -> List[Metric]:
    metrics_functions = []
    for metric in metrics:
        if isinstance(metric, str):
            metric = {'name': metric}

        metric_name = metric['name']

        f = get_metric_by_name(metric_name)

        inputs = metric.get('inputs', in_y + out_vars)
        if isinstance(inputs, str):
            inputs = [inputs]

        metrics_functions.append(Metric(metric_name, f, inputs))
    return metrics_functions


def prettify_metrics(metrics: List[Tuple[str, float]], precision: int = 4) -> OrderedDict:
    """Prettifies the dictionary of metrics."""
    prettified_metrics = OrderedDict()
    for key, value in metrics:
        value = round(value, precision)
        prettified_metrics[key] = value
    return prettified_metrics
