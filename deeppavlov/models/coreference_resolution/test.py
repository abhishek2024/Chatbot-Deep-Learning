from deeppavlov.models.coreference_resolution.kpi_model import CorefModel
from deeppavlov.models.coreference_resolution.coreference_iterator import CorefIterator
from deeppavlov.models.coreference_resolution.coreference_reader import CorefReader


config_model = {"save_path": "./some_folder/checkponit.indxsg",
                "load_path": "./some_folder/checkponit.indxsg"}

# "./some_folder/checkponit.indxsg"

model = CorefModel(**config_model)

print("Fuck yea")
