# %%
# %load_ext autoreload
# %autoreload 2

# %%
from deeppavlov import configs
from deeppavlov import build_model
from deeppavlov.agents.default_agent.default_agent import DefaultAgent
# %%
# deeppavlov/configs/dp_assistant/agent_transformer_chit_chat_40k_v01_1_20.json
# model = build_model(config=configs.generative_chit_chat.transformer_chit_chat)
model = build_model(config=configs.dp_assistant.agent_transformer_chit_chat_40k_v01_1_20)
# model = build_model(config=configs.generative_chit_chat.transformer_chit_chat_100k)
# model = build_model(config=configs.generative_chit_chat.transformer_chit_chat)
model.load()
# %%
# model(['Как дела?', 'Все хорошо, спасибо!', 'Что делаешь?'], [['Привет!', 'Добрый день!'], ['Ты как?'], []], [])

# # %%
# agent = DefaultAgent([model])
# while True:
#     utterance = input("::")
#     print(agent([utterance]))

# %%
dialogs = [
    {
      "id": "5c65706b0110b377e17eba41",
      "location": None,
      "utterances": [
        {
          "id": "5c65706b0110b377e17eba39",
          "text": "Привет!",
          "user_id": "5c65706b0110b377e17eba37",
          "annotations": {
            "ner": {
              "tokens": [
                "Привет",
                "!"
              ],
              "tags": [
                "O",
                "O"
              ]
            },
            "coref": [
            ],
            "sentiment": "speech"
          },
          "date_time": "2019-02-14 13:43:07.594000"
        },
        {
          "id": "5c65706b0110b377e17eba3a",
          "active_skill": "chitchat",
          "confidence": 0.85,
          "text": "Привет, я бот!",
          "user_id": "5c65706b0110b377e17eba38",
          "annotations": {
            "ner": {
              "tokens": [
                "Привет",
                ",",
                "я",
                "бот",
                "!"
              ],
              "tags": [
                "O",
                "O",
                "O",
                "O",
                "O"
              ]
            },
            "coref": [
            ],
            "sentiment": "speech"
          },
          "date_time": "2019-02-14 13:43:07.595000"
        },
        {
          "id": "5c65706b0110b377e17eba3b",
          "text": "Как дела?",
          "user_id": "5c65706b0110b377e17eba37",
          "annotations": {
            "ner": {
              "tokens": [
                "Как",
                "дела",
                "?"
              ],
              "tags": [
                "O",
                "O",
                "O"
              ]
            },
            "coref": [
            ],
            "sentiment": "speech"
          },
          "date_time": "2019-02-14 13:43:07.595000"
        },
        {
          "id": "5c65706b0110b377e17eba3c",
          "active_skill": "chitchat",
          "confidence": 0.9333,
          "text": "Хорошо, а у тебя как?",
          "user_id": "5c65706b0110b377e17eba38",
          "annotations": {
            "ner": {
              "tokens": [
                "Хорошо",
                ",",
                "а",
                "у",
                "тебя",
                "как",
                "?"
              ],
              "tags": [
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O"
              ]
            },
            "coref": [
            ],
            "sentiment": "speech"
          },
          "date_time": "2019-02-14 13:43:07.595000"
        },
        {
          "id": "5c65706b0110b377e17eba3d",
          "text": "И у меня нормально. Когда родился Петр Первый?",
          "user_id": "5c65706b0110b377e17eba37",
          "annotations": {
            "ner": {
              "tokens": [
                "И",
                "у",
                "меня",
                "нормально",
                ".",
                "Когда",
                "родился",
                "Петр",
                "Первый",
                "?"
              ],
              "tags": [
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "B-PER",
                "I-PER",
                "O"
              ]
            },
            "coref": [
            ],
            "sentiment": "neutral"
          },
          "date_time": "2019-02-14 13:43:07.595000"
        },
        {
          "id": "5c65706b0110b377e17eba3e",
          "active_skill": "odqa",
          "confidence": 0.74,
          "text": "в 1672 году",
          "user_id": "5c65706b0110b377e17eba38",
          "annotations": {
            "ner": {
              "tokens": [
                "в",
                "1672",
                "году"
              ],
              "tags": [
                "O",
                "O",
                "O"
              ]
            },
            "coref": [
            ],
            "sentiment": "neutral"
          },
          "date_time": "2019-02-14 13:43:07.595000"
        },
        {
          "id": "5c65706b0110b377e17eba3f",
          "text": "спасибо",
          "user_id": "5c65706b0110b377e17eba37",
          "annotations": {
            "ner": {
              "tokens": [
                "спасибо"
              ],
              "tags": [
                "O"
              ]
            },
            "coref": [
            ],
            "sentiment": "speech"
          },
          "date_time": "2019-02-14 13:43:07.595000"
        }
      ],
      "user": {
        "id": "5c65706b0110b377e17eba37",
        "user_telegram_id": "0801e781-0b76-43fa-9002-fcdc147d35af",
        "user_type": "human",
        "device_type": None,
        "personality": None
      },
      "bot": {
        "id": "5c65706b0110b377e17eba38",
        "user_type": "bot",
        "persona": [
            "Мне нравится общаться с людьми",
            "Пару лет назад я окончила вуз с отличием",
            "Я работаю в банке",
            "В свободное время помогаю пожилым людям в благотворительном фонде",
            "Люблю путешествовать",
        ]
      },
      "channel_type": "telegram"
    },
    {
      "id": "5c65706b0110b377e17eba47",
      "location": None,
      "utterances": [
        {
          "id": "5c65706b0110b377e17eba43",
          "text": "Когда началась Вторая Мировая?",
          "user_id": "5c65706b0110b377e17eba37",
          "annotations": {
            "ner": {
              "tokens": [
                "Когда",
                "началась",
                "Вторая",
                "Мировая",
                "?"
              ],
              "tags": [
                "O",
                "O",
                "O",
                "O",
                "O"
              ]
            },
            "coref": [
            ],
            "sentiment": "neutral"
          },
          "date_time": "2019-02-14 13:43:07.601000"
        },
        {
          "id": "5c65706b0110b377e17eba44",
          "active_skill": "odqa",
          "confidence": 0.99,
          "text": "1939",
          "user_id": "5c65706b0110b377e17eba38",
          "annotations": {
            "ner": {
              "tokens": [
                "1939"
              ],
              "tags": [
                "O"
              ]
            },
            "coref": [
            ],
            "sentiment": "neutral"
          },
          "date_time": "2019-02-14 13:43:07.601000"
        },
        {
          "id": "5c65706b0110b377e17eba45",
          "text": "Спасибо, бот!",
          "user_id": "5c65706b0110b377e17eba37",
          "annotations": {
            "ner": {
              "tokens": [
                "Спасибо",
                ",",
                "бот",
                "!"
              ],
              "tags": [
                "O",
                "O",
                "O",
                "O"
              ]
            },
            "coref": [
            ],
            "sentiment": "speech"
          },
          "date_time": "2019-02-14 13:43:07.601000"
        }
      ],
      "user": {
        "id": "5c65706b0110b377e17eba42",
        "user_telegram_id": "a27a94b6-2b9d-4802-8eb6-6581e6f8cd8c",
        "user_type": "human",
        "device_type": None,
        "personality": None
      },
      "bot": {
        "id": "5c65706b0110b377e17eba38",
        "user_type": "bot",
        "persona": [
            "Мне нравится общаться с людьми",
            "Пару лет назад я окончила вуз с отличием",
            "Я работаю в банке",
            "В свободное время помогаю пожилым людям в благотворительном фонде",
            "Люблю путешествовать",
        ]
      },
      "channel_type": "telegram"
    }
  ]
# %%
model(dialogs)