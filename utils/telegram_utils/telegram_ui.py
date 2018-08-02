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

from pathlib import Path

import telebot

from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config

from utils.telegram_utils.telegram_profile import *

TELEGRAM_UI_CONFIG_FILENAME = 'models_info.json'

download_path = (Path(__file__) / ".." / ".." / ".." / "download").resolve()

def init_bot_for_model(token, model, config):

    bot = telebot.TeleBot(token)

    config_dir = Path(__file__).resolve().parent
    config_path = Path(config_dir, TELEGRAM_UI_CONFIG_FILENAME).resolve()
    models_info = read_json(str(config_path))

    model_name = type(model.get_main_component()).__name__
    model_info = models_info[model_name] if model_name in models_info else models_info['@default']

    @bot.message_handler(commands=['start'])
    def send_start_message(message):
        chat_id = message.chat.id
        out_message = model_info['start_message']
        if hasattr(model, 'reset'):
            model.reset()
        bot.send_message(chat_id, out_message)

    @bot.message_handler(commands=['help'])
    def send_help_message(message):
        chat_id = message.chat.id
        out_message = model_info['help_message']
        bot.send_message(chat_id, out_message)


    if 'telegram-bot' in config:
        path2db = config['telegram-bot'].get('path2db')
        path2db = download_path / path2db
        db_conn = connection_init(path2db)
    else:
        db_conn = None
    if db_conn:
        @bot.message_handler()
        def handle_inference(message):
            chat_id = message.chat.id
            context = message.text

            profile = get_profile(db_conn, chat_id)
            profile['history'].append(context)

            history = " ".join(profile['history'])

            pred = model([[history, context, message]])
            reply_message = str(pred[0])

            profile['history'].append(reply_message)
            update_profile(db_conn, chat_id, profile)

            bot.send_message(chat_id, reply_message)
    else:
        @bot.message_handler()
        def handle_inference(message):
            chat_id = message.chat.id
            context = message.text

            pred = model([context])
            reply_message = str(pred[0])
            bot.send_message(chat_id, reply_message)

    bot.polling(none_stop=True)


def interact_model_by_telegram(config_path, token):
    config = read_json(config_path)

    model = build_model_from_config(config)

    init_bot_for_model(token, model, config)
