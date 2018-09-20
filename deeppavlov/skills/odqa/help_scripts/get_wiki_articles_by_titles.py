"""
Find and output all Wikipedia articles from SQuAD dataset.
Input SQuAD dev or train set prepared for ranker.
Output in format: {"question": q, "title": t, "text": txt}, where q, t, txt are string data for a single instance.
"""

import argparse
import unicodedata
from typing import List, Dict, Union
import json
import sqlite3
from copy import copy

parser = argparse.ArgumentParser()

parser.add_argument("-squad_data_path",
                    help="path to a SQuAD dataset prepared for ranker", type=str,
                    default="/media/olga/Data/datasets/squad/original/dev-v1.1.json")
parser.add_argument("-db_path", help="local path or URL to a SQLite DB with Wikipedia articles",
                    type=str,
                    default="/media/olga/Data/datasets/drqa/wikipedia.db")
parser.add_argument("-output_path", help="path to an output JSON file", type=str,
                    default="/media/olga/Data/datasets/squad/wiki_articles/drqa_squad_dev_articles.json")

NEW2OLD_WIKI_TITLES = {  # new titles are keys, old are values
    'Sky UK': 'Sky (United Kingdom)',
    'Financial crisis of 2007–2008': 'Financial crisis of 2007–08',
    'Huguenots': 'Huguenot',
    'Phonograph record': 'Gramophone record',
    'Sony Music': 'Sony Music Entertainment',
    'Mary, mother of Jesus': 'Mary (mother of Jesus)',
    'Multiracial Americans': 'Multiracial American',
    'Endangered Species Act of 1973': 'Endangered Species Act',
    'Cardinal (Catholic Church)': 'Cardinal (Catholicism)',
    'Universal Pictures': 'Universal Studios',
    'Bill and Melinda Gates Foundation': 'Bill & Melinda Gates Foundation',
    'Gates Foundation': 'Bill & Melinda Gates Foundation',
    'Videotelephony': 'Videoconferencing',
    'Imamah (Shia)': 'Imamah (Shia doctrine)',
    'Seven Years\' War': 'Seven_Years%27_War',
    'Molotov–Ribbentrop Pact': 'Molotov%E2%80%93Ribbentrop_Pact',
    # 'Bill and Melinda Gates Foundation': 'Bill_%26_Melinda_Gates_Foundation',
    'Brasília': 'Bras%C3%ADlia',
    'Kievan_Rus\'': 'Kievan_Rus%27',
    'St. John\'s, Newfoundland and Labrador': 'St._John%27s,_Newfoundland_and_Labrador',
    'Saint Barthélemy': 'Saint_Barth%C3%A9lemy',
}

NEW2OLD_WIKI_TITLES_TRAIN = {  # new titles are keys, old are values
    'Sky UK': 'Sky (United Kingdom)',
    'Huguenots': 'Huguenot',
    'Phonograph record': 'Gramophone record',
    'Sony Music': 'Sony Music Entertainment',
    'Mary, mother of Jesus': 'Mary (mother of Jesus)',
    'Multiracial Americans': 'Multiracial American',
    'Endangered Species Act of 1973': 'Endangered Species Act',
    'Cardinal (Catholic Church)': 'Cardinal (Catholicism)',
    'Universal Pictures': 'Universal Studios',
    # 'Bill and Melinda Gates Foundation': 'Bill & Melinda Gates Foundation',
    'Gates Foundation': 'Bill & Melinda Gates Foundation',
    'Videotelephony': 'Videoconferencing',
    'Imamah (Shia)': 'Imamah (Shia doctrine)',
    'Seven Years\' War': 'Seven_Years%27_War',
    'Molotov–Ribbentrop Pact': 'Molotov%E2%80%93Ribbentrop_Pact',
    'Bill & Melinda Gates Foundation': 'Bill_%26_Melinda_Gates_Foundation',
    'Brasília': 'Bras%C3%ADlia',
    'Kievan Rus\'': 'Kievan_Rus%27',
    'St. John\'s, Newfoundland and Labrador': 'St._John%27s,_Newfoundland_and_Labrador',
    'Saint Barthélemy': 'Saint_Barth%C3%A9lemy',
    'Financial crisis of 2007–2008': 'Financial_crisis_of_2007%E2%80%9308',
    'Jehovah\'s Witnesses': 'Jehovah%27s_Witnesses'
}

SQUAD2DRQA_TITLES = {
    'Kievan_Rus%27': 'Kievan Rus\'',
    'Jehovah%27s_Witnesses': 'Jehovah\'s Witnesses',
    'Bill_%26_Melinda_Gates_Foundation': 'Bill & Melinda Gates Foundation',
    'St._John%27s,_Newfoundland_and_Labrador': 'St. John\'s, Newfoundland and Labrador',
    'Saint_Barth%C3%A9lemy': 'Saint Barthélemy',
    'Bras%C3%ADlia': 'Brasília',
    'Seven_Years%27_War': 'Seven Years\' War',
    'Molotov%E2%80%93Ribbentrop_Pact': 'Molotov–Ribbentrop Pact',
    'Sony_Music_Entertainment': 'Sony Music',
    'Multiracial_American': 'Multiracial Americans',
    'Cardinal_(Catholicism)': 'Cardinal (Catholic Church)',
    'Endangered_Species_Act': 'Endangered Species Act of 1973',
    'Financial_crisis_of_2007%E2%80%9308': 'Financial crisis of 2007–2008',
    'Mary_(mother_of_Jesus)': 'Mary, mother of Jesus',
    'Sky_(United_Kingdom)': 'Sky UK'
}


def encode_utf8(s: str):
    return unicodedata.normalize('NFD', s).encode('utf-8')


def read_json(fpath):
    with open(fpath) as fin:
        return json.load(fin)


def write_json(data, fpath):
    with open(fpath, 'w') as fout:
        json.dump(data, fout)


def extract_squad_orig_titles(squad_path):
    dataset = read_json(squad_path)
    titles = set(article['title'] for article in dataset['data'])
    return titles


def search_titles_in_db(db_path, squad_titles):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    extracted = []

    for title in squad_titles:

        text = ''

        orig_title = copy(title)

        if title in NEW2OLD_WIKI_TITLES_TRAIN.values():
            title = [k for k in NEW2OLD_WIKI_TITLES_TRAIN.keys() if NEW2OLD_WIKI_TITLES_TRAIN[k] == title][0]

        elif title in NEW2OLD_WIKI_TITLES.values():
            title = \
            [k for k in NEW2OLD_WIKI_TITLES.keys() if NEW2OLD_WIKI_TITLES[k] == title][
                0]

        elif title.replace('_', ' ') in NEW2OLD_WIKI_TITLES.values():
            title = \
                [k for k in NEW2OLD_WIKI_TITLES.keys() if NEW2OLD_WIKI_TITLES[k] == title.replace(
                    '_', ' ')][0]

        elif title.replace('_', ' ') in NEW2OLD_WIKI_TITLES_TRAIN.values():
            title = \
                [k for k in NEW2OLD_WIKI_TITLES_TRAIN.keys() if NEW2OLD_WIKI_TITLES_TRAIN[k] == title.replace(
                    '_', ' ')][0]

        elif title.replace('_', ' ') in NEW2OLD_WIKI_TITLES_TRAIN.keys():
            title = NEW2OLD_WIKI_TITLES_TRAIN[title.replace('_', ' ')]

        else:
            title = title.replace('_', ' ')


        # DrQa
        # if title in SQUAD2DRQA_TITLES.keys():
        #     title = SQUAD2DRQA_TITLES[title]
        #
        # title = title.replace('_', ' ')



        try:
            c.execute(
                "SELECT text FROM documents WHERE id = ?",
                (title,)
            )
            result = c.fetchone()
            text = result[0]
        except Exception:
            try:
                title = unicodedata.normalize('NFD', title)
                c.execute(
                    "SELECT text FROM documents WHERE id = ?",
                    (title,)
                )
                result = c.fetchone()
                text = result[0]
            except Exception:
                # DrQa
                # if title != 'List of numbered streets in Manhattan':
                print(f"Exception while fetching title {title}")
                print(f"Original title: {orig_title}")
                text = ''

        # DrQa
        # if title == 'List of numbered streets in Manhattan':
        #     new_data = read_json('/media/olga/Data/datasets/squad/wiki_articles/squad_dev_articles.json')
        #     for article_dict in new_data:
        #         if article_dict['title'] == 'List_of_numbered_streets_in_Manhattan':
        #             text = article_dict['text']

        assert title
        assert text

        extracted.append({'title': orig_title, 'text': text})

    return extracted


def main():
    args = parser.parse_args()

    squad_titles = extract_squad_orig_titles(args.squad_data_path)
    titles_with_text = search_titles_in_db(args.db_path, squad_titles)

    write_json(titles_with_text, args.output_path)

    print('Done!')


if __name__ == "__main__":
    main()