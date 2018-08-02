

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

logger = get_logger(__name__)

def_persona = [
                'i love to meet new people.',
                'i have a turtle named timothy.',
                'my favorite sport is ultimate frisbee.',
                'my parents are living in bora bora.',
                'my parents are living in bora bora.',
                ]


@register('personachat_tg_in')
class PersonaChatTokenizer(Component):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, history, context, message, **kwargs):
        """ Prepares data for pipeline

        Args:
            history: list of str
            context: str
            message: metadata of tg

        Returns:
            persona: str
            dialog_history: str
            x_utterance: str
        """

        persona = [' '.join(def_persona)]
        dialog_history =  [' '.join(history[-2:])]
        x_utterance =  context
        return persona, dialog_history, x_utterance
