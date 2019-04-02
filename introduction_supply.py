# %%
# %load_ext autoreload
# %autoreload 2

# %%
from deeppavlov import configs
from deeppavlov import build_model
from deeppavlov.agents.default_agent.default_agent import DefaultAgent
import json
import pathlib
import tqdm
from nltk import word_tokenize
import collections
import numpy as np
import pprint
# %%
# model = build_model(config=configs.generative_chit_chat.transformer_chit_chat)
model = build_model(config=configs.generative_chit_chat.transformer_chit_chat_40k)
# model = build_model(config=configs.generative_chit_chat.transformer_chit_chat_100k)
# model = build_model(config=configs.generative_chit_chat.transformer_chit_chat)
model.load()
# %%
agent = DefaultAgent([model])
utters = json.load(pathlib.Path('utterances_sushi.json').open())['counterparts']
sushi = json.load(pathlib.Path('utterances_sushi.json').open())['sushi']
# %%
answers = []
for ut in tqdm.tqdm(utters):
    answers.append(agent([ut]))
answers = [a[-1] for a in answers if a]
json.dump(answers, pathlib.Path('answers_sushi_40.json').open('wt'), ensure_ascii=False)

# %%
# %%
# answers = json.load(pathlib.Path('answers_sushi.json').open())
# %%
def count(utters):
    counters = [collections.Counter(word_tokenize(ut)) for ut in utters]
    counter = sum(counters, collections.Counter())
    return counter
def get_stats(values):
    stats = {}
    stats['values'] = np.array(list(values))
    stats['values_n'] = stats['values'].shape[-1]
    stats['mean'] = np.mean(stats['values'])
    stats['std'] = np.std(stats['values'])
    stats['median'] = np.median(stats['values'])
    stats['percentile_25'] = np.percentile(stats['values'], 25)
    stats['percentile_75'] = np.percentile(stats['values'], 75)
    return stats
answers_stats = get_stats(count(answers).values())
sushi_stats = get_stats(count(sushi).values())
pprint.pprint([answers_stats, sushi_stats])
# %%
agent = DefaultAgent([model])
while True:
    utterance = input("::")
    print(agent([utterance]))