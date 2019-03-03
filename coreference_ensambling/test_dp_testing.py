from pathlib import Path

from coreference_ensambling.model_calling import test_model_
from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.commands.train import _parse_metrics, get_iterator_from_config, read_data_by_config
from deeppavlov.core.commands.utils import parse_config

model_config = Path("/home/mks/projects/DeepPavlov/deeppavlov/configs/coreference_resolution/test_ensamble/kfold_ens/gold_elmo2_fold_1.json")
config = parse_config(model_config)
model = build_model(config)
train_config = config.get("train")

in_y = config['chainer'].get('in_y', ['y'])
if isinstance(in_y, str):
    in_y = [in_y]
if isinstance(config['chainer']['out'], str):
    config['chainer']['out'] = [config['chainer']['out']]
metrics_functions = _parse_metrics(train_config['metrics'], in_y, config['chainer']['out'])

data = read_data_by_config(config)
iterator = get_iterator_from_config(config, data)
# calculate metrics
res = test_model_(model, metrics_functions, iterator, train_config.get('batch_size', -1), 'test', False)
print(res['metrics'])
