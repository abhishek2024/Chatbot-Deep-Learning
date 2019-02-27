# %%
# %load_ext autoreload
# %autoreload 2

# %%
from deeppavlov import configs
from deeppavlov import build_model
from deeppavlov.agents.default_agent.default_agent import DefaultAgent
# %%
model = build_model(config=configs.generative_chit_chat.transformer_chit_chat)
# model = build_model(config=configs.generative_chit_chat.transformer_chit_chat)
model.load()
# %%
# model(['Как дела?', 'Все хорошо, спасибо!', 'Что делаешь?'], [['Привет!', 'Добрый день!'], ['Ты как?'], []], [])

# %%
agent = DefaultAgent([model])
while True:
    utterance = input("::")
    print(agent([utterance]))