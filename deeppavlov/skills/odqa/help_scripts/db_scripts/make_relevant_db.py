import argparse
import sqlite3
import unicodedata
from typing import List, Union
from itertools import chain

from nltk.tokenize import sent_tokenize

from deeppavlov.core.common.file import read_json
from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator

parser = argparse.ArgumentParser()

parser.add_argument("-input_db_path",
                    help="path to an input db with wikipedia articles \
                    (BETTER MAKE A COPY SINCE IT WILL BE CHANGED IN RUNTIME)",
                    type=str,
                    default="/media/olga/Data/projects/DeepPavlov/download/odqa/enwiki_cut_irrelevant.db")
parser.add_argument("-ids_path", help="path to a JSON dataset with ids", type=str,
                    default="/media/olga/Data/projects/DeepPavlov/deeppavlov/skills/odqa/help_scripts/db_scripts/relevant_ids.json")
parser.add_argument("-iterator_db", help="path to a JSON dataset with ids", type=str,
                    default="/media/olga/Data/projects/DeepPavlov/download/odqa/enwiki_cut_truth.db")


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
        'Jehovah\'s Witnesses': 'Jehovah%27s_Witnesses',
        'Beyoncé': 'Beyonc%C3%A9'
    }


def encode_utf8(s: str):
    return unicodedata.normalize('NFD', s)


def main():
    args = parser.parse_args()
    db_path = args.input_db_path
    saved_ids = read_json(args.ids_path)
    rel_ids = set(saved_ids).union(set(NEW2OLD_WIKI_TITLES_TRAIN.keys()))
    rel_ids = rel_ids.union(NEW2OLD_WIKI_TITLES_TRAIN.values())
    rel_ids = rel_ids.union(set([encode_utf8(k) for k in NEW2OLD_WIKI_TITLES_TRAIN.keys()]))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    iterator = SQLiteDataIterator(args.iterator_db)

    count = 0
    for i in iterator.doc_ids:
        if i not in rel_ids:
            cursor.execute("DELETE FROM documents WHERE id=?", (i,))
        count += 1
        print(f"Processing id {count}")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
