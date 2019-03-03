import datetime
import time
from collections import OrderedDict, namedtuple
from typing import Dict, List, Tuple, Union

from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator

Metric = namedtuple('Metric', ['name', 'fn', 'inputs'])


def prettify_metrics(metrics: List[Tuple[str, float]], precision: int = 4) -> OrderedDict:
    """Prettifies the dictionary of metrics."""
    prettified_metrics = OrderedDict()
    for key, value in metrics:
        value = round(value, precision)
        prettified_metrics[key] = value
    return prettified_metrics


def test_model_(model: Chainer, metrics_functions: List[Metric],
                iterator: DataLearningIterator, batch_size=-1, data_type='valid',
                start_time: float = None) -> Dict[str, Union[int, OrderedDict, str]]:
    expected_outputs = list(set().union(model.out_params, *[m.inputs for m in metrics_functions]))

    outputs = {out: [] for out in expected_outputs}
    examples = 0
    for x, y_true in iterator.gen_batches(batch_size, data_type, shuffle=False):
        examples += len(x)
        y_predicted = list(model.compute(list(x), list(y_true), targets=expected_outputs))
        if len(expected_outputs) == 1:
            y_predicted = [y_predicted]
        for out, val in zip(outputs.values(), y_predicted):
            out += list(val)

    metrics = [(m.name, m.fn(*[outputs[i] for i in m.inputs])) for m in metrics_functions]

    report = {
        'eval_examples_count': examples,
        'metrics': prettify_metrics(metrics),
        'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time + 0.5)))
    }
    return report
