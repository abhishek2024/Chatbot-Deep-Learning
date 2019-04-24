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

from logging import getLogger
import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component

logger = getLogger(__name__)


@register("chitchat_reranker")
class ChitChatReRanker(Component):

    def __init__(self,
                 num_context_turns: int = 10,
                 lambda_coeff: float = 10.0,
                 **kwargs):
        self.num_context_turns = num_context_turns
        self.lambda_coeff = lambda_coeff

    def __call__(self,
                 top_responses_batch,
                 top_scores_batch,
                 sentiment_batch,
                 intents_batch):
        """
        intents are: "declarative", "interrogative", "imperative"
        sentiment are: "neutral", "positive", "skip", "speech", "negative"

        Returns:
            batch of the best responses
        """

        # print("[top_responses_batch]: ", top_responses_batch)

        # reshape sentiment_batch
        # reshape intents_batch
        batch_size = self.num_context_turns + len(top_responses_batch[0])
        print("[batch_size]: ", batch_size)
        reshaped_sentiment = []
        reshaped_intents = []
        for i in range(len(sentiment_batch) // batch_size):
            reshaped_sentiment.append(np.array(sentiment_batch[i*batch_size: (i+1)*batch_size]).squeeze().tolist())
            reshaped_intents.append(np.array(intents_batch[i*batch_size: (i+1)*batch_size]).squeeze().tolist())

        responses_batch = []
        responses_preds = []

        for i in range(len(top_responses_batch)):
            # heuristics...

            # filter by intents, len of intents_batch[i]: self.num_context_turns + len(top_responses_batch[i])
            last_utter_intent = reshaped_intents[i][self.num_context_turns - 1]
            responses_intents = reshaped_intents[i][self.num_context_turns:]

            print("len, [resp intents]:", len(responses_intents), [(k, v) for k, v in zip(responses_intents, top_responses_batch[0])])


            filtered_intents_idx = [j for j in range(len(top_responses_batch[i]))]
            if last_utter_intent == "interrogative":
                filtered_intents_idx = [j for j in range(len(top_responses_batch[i])) if responses_intents[j] in ["declarative"]]
            elif last_utter_intent == "imperative":
                filtered_intents_idx = [j for j in range(len(top_responses_batch[i])) if responses_intents[j] in ["interrogative", "declarative"]]

            # filter by sentiment
            # negative -> neutral, neutral -> positive
            last_utter_sentiment = reshaped_sentiment[i][self.num_context_turns - 1]
            responses_sentiment = reshaped_sentiment[i][self.num_context_turns:]

            print("len, [resp sentiment]:", len(responses_sentiment), [(k, v) for k, v in zip(responses_sentiment, top_responses_batch[0])])

            filtered_sentiment_idx = [j for j in range(len(top_responses_batch[i]))  if
                                      responses_sentiment[j] not in ["negative", "skip"]]
            if last_utter_sentiment == "negative":
                filtered_sentiment_idx = [j for j in range(len(top_responses_batch[i])) if
                                          responses_sentiment[j] not in ["negative", "skip"]]
            elif last_utter_sentiment == "neutral":
                filtered_sentiment_idx = [j for j in range(len(top_responses_batch[i])) if
                                          responses_sentiment[j] in ["neutral", "positive"]]  # may be positive only,   not ["neutral", "positive"]
            elif last_utter_sentiment == "speech":
                filtered_sentiment_idx = [j for j in range(len(top_responses_batch[i])) if
                                          responses_sentiment[j] == "speech"]
            elif last_utter_sentiment == "positive":
                filtered_sentiment_idx = [j for j in range(len(top_responses_batch[i])) if
                                          responses_sentiment[j] == "positive"]

            print("[debug]: \intents/ [last]", last_utter_intent, "[idx]", filtered_intents_idx,
                  "|| \sentiment/ [last]", last_utter_sentiment, "[idx]", filtered_sentiment_idx)

            # Special cases
            # 1. Negative + imperative -> sad :-(
            if last_utter_intent == "imperative" and last_utter_sentiment == "negative":
                filtered_intents_idx = [j for j in range(len(top_responses_batch[i])) if responses_intents[j] in ["interrogative"]]

            # Format output
            idx = sorted(list(set(filtered_intents_idx).intersection(set(filtered_sentiment_idx))))
            print("[idx]: ", idx)
            if len(idx) > 0:
                # use sentiment + intents
                pass
            elif len(filtered_sentiment_idx):
                # use sentiment only
                idx = filtered_sentiment_idx
            elif len(filtered_intents_idx):
                # use intents only
                idx = filtered_intents_idx
            else:
                # fallback, use flat candidates as are
                idx = [k for k in range(len(top_responses_batch[i]))]

            candidates = [top_responses_batch[i][j] for j in idx]
            scores = [top_scores_batch[i][j] for j in idx]

            print("[candidates]: ", candidates)

            pcount = np.arange(len(candidates))
            w = np.exp(-pcount / self.lambda_coeff)
            w = w / w.sum()
            # print("distribution:", w)  # DEBUG

            chosen_index = np.random.choice(idx, p=w)
            print("[chosen_index]", chosen_index)
            print("[answer]: ", top_responses_batch[i][chosen_index])

            to_output = top_responses_batch[i][chosen_index]
            if last_utter_sentiment in ["negative"]:
                to_output += " 😔 ..."
            if responses_sentiment[chosen_index] in ["positive"]:
                if np.random.random() < 0.5:          # append smile with prob=0.5 when the answer is positive
                    to_output += " 😊"

            responses_batch.append(to_output)
            responses_preds.append(top_scores_batch[i][chosen_index])

        return responses_batch, responses_preds