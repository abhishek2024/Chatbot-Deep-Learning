from deeppavlov.models.coreference_resolution.kpi_model import CorefModel
from deeppavlov.models.coreference_resolution.coreference_iterator import CorefIterator
from deeppavlov.models.coreference_resolution.coreference_reader import CorefReader

reader = CorefReader()
dataset = reader.read(data_path="/home/mks/projects/DeepPavlov/download/rucor_conll")
iterator = CorefIterator(data=dataset)

config_model = {"save_path": "./checkpoints/",
                "load_path": "./checkpoints/",
                "model_file": "./",
                "embedding_size": 300,
                "emb_format": "bin",
                "train_on_gold": True}

model = CorefModel(**config_model)
