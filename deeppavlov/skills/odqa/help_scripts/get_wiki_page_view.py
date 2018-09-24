import requests
import unicodedata
import json
from urllib import parse

from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator
from deeppavlov.core.common.file import save_json, read_json

DB_PATH = '/media/olga/Data/projects/DeepPavlov/download/odqa/enwiki.db'
URL = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents/{}/monthly/20171105/20181105'
SAVE_PATH = 'popularities.json'
# headers = {
#     'Content-Type': 'application/json',
# }

iterator = SQLiteDataIterator(load_path=DB_PATH)
doc_ids = iterator.get_doc_ids()
del iterator
title2popularity = read_json(SAVE_PATH)

for orig_title in doc_ids:
    try:
        try:
            title2popularity[orig_title]
            continue
        except KeyError:
            pass

        title = orig_title.replace(' ', '_')
        title = title.replace("%", "%25")
        # title = title.replace("/", "////")
        title = title.replace('&amp;', '%26')
        title = title.replace('&quot;', '\"')
        title = title.replace("?", "%3F")
        title = title.replace("\'", "%27")
        title = unicodedata.normalize('NFC', title)
        title = title.replace("è:", "è:")
        title = title.replace("ù_", "ù_")
        title = title.replace("ü", "ü")
        title = title.replace("ī", "ī")
        title = title.replace("+", '%2B')


        if orig_title == "&quot;Chūsotsu&quot; &quot;Chūkara&quot;":
            title = '\"Chūsotsu\" \"Chūkara\"'
        elif orig_title == "&quot;The Spaghetti Incident?&quot;":
            title = "The_Spaghetti_Incident%3F"
        elif orig_title == "&quot;Üns&quot; Theatre":
            title = "\"Üns\"_Theatre"
        elif orig_title == ".40 S&amp;W":
            title = ".40_S%26W"
        elif orig_title == "1/2 &amp; 1/2":
            title = "1/2_&_1/2"
        elif orig_title == "1/2 + 1/4 + 1/8 + 1/16 + ⋯":
            title = "1/2 + 1/4 + 1/8 + 1/16 + ⋯"
        # elif orig_title == "+/- (band)":
        #     title = "%2B/-_(band)"
        # elif orig_title == "(I Wish I Knew How It Would Feel to Be) Free / One":
        #     title = "(I_Wish_I_Knew_How_It_Would_Feel_to_Be)_Free/One"
        # elif orig_title == "\'Till I Get My Way/Girl Is on My Mind":
        #     title = "\'Till_I_Get_My_Way_/_Girl_Is_on_My_Mind"

        # title = parse.quote_plus(title)

        url = URL.format(title)
        response = requests.get(url).json()
        try:
            popularities = [item['views'] for item in response['items']]
        except KeyError:
            title = parse.quote_plus(title)
            url = URL.format(title)
            response = requests.get(url).json()
            popularities = [item['views'] for item in response['items']]
        popularity = sum(popularities) / len(popularities)
        title2popularity[orig_title] = popularity
    except KeyError:
        save_json(title2popularity, SAVE_PATH)
        print(orig_title)
        print(title)
        pass
    except json.decoder.JSONDecodeError:
        save_json(title2popularity, SAVE_PATH)
        print(orig_title)
        print(title)
        pass

save_json(title2popularity, SAVE_PATH)

