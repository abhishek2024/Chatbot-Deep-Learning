import numpy as np
from typing import List, Iterable, Callable, Union

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.component import Component
from deeppavlov.models.ranking.keras_siamese_model import SiameseModel
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from deeppavlov.core.common.registry import register

log = get_logger(__name__)

@register('siamese_vectorizer')
class SiameseVectorizer(Component):
    """The class for getting context vectors from siamese models"""

    def __init__(self, model: SiameseModel, *args, **kwargs) -> None:
        super().__init__()

        if model.attention:
            log.error(f"{model} has attention herefore it does not have "
                      "context model for vectorization")
            raise ValueError(model)
        self.model = model
    
    def __call__(self, batch):
        batch = self.model._make_batch(list(batch))
        return self.model._predict_context_on_batch(batch)

    def reset(self) -> None:
        pass

    def process_event(self) -> None:
        pass
