# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, softwaredata
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import re
from typing import Dict, List, Tuple

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register("mltrack_reader")
class MLtrackReader(DatasetReader):
    """The class to read the MLtrack dataset from files.

    Please, see https://contest.yandex.ru/algorithm2018
    """

    def read(self,
             data_path: str,
             seed: int = None, *args, **kwargs) -> Dict[str, List[Tuple[List[str], int]]]:
        """Read the MLtrack dataset from files.

        Args:
            data_path: A path to a folder with dataset files.
            seed: Random seed.
        """

        data_path = expand_path(data_path)
        train_fname = data_path / "train.tsv"
        test_fname = data_path / "final.tsv"
        train_data = self.build_data(train_fname, "train")
        valid_data = self.build_data(train_fname, "valid")
        test_data = self.build_data(test_fname, "test")
        dataset = {"train": train_data, "valid": valid_data, "test": test_data}
        return dataset

    def build_data(self, name, data_type):
        lines = []
        with open(name, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            for line in reader:
                lines.append(line)
        texts, labels = [], []
        if data_type == "test":
            for line in lines:
                line.extend(["good", 1.])
        y_voc = {"good": 2, "neutral": 1, "bad": 0}
        if data_type != "train":
            pad = [["", "", "0"]]
            cur_id = lines[0][0]
            num = 0
            for line in lines:
                context_id = line[0]
                if context_id == cur_id:
                    num += 1
                    context = self.clean_data(" ".join(line[1:4]))
                    reply = self.clean_data(line[5])
                    texts.append([context, reply, context_id])
                    labels.append(y_voc[line[-2]])
                else:
                    texts += (6-num)*pad
                    labels += (6-num)*[0]
                    num = 1
                    cur_id = line[0]
                    context = self.clean_data(" ".join(line[1:4]))
                    reply = self.clean_data(line[5])
                    texts.append([context, reply, context_id])
                    labels.append(y_voc[line[-2]])
        else:
            for line in lines:
                context_id = line[0]
                context = self.clean_data(" ".join(line[1:4]))
                reply = self.clean_data(line[5])
                texts.append([context, reply, context_id])
                labels.append(y_voc[line[-2]])
        return list(zip(texts, labels))

    def clean_data(self, text):
        """Clean or fix invalid symbols in given text"""

        def clean(string):
            str1 = re.sub(
                '{ c : \$00ffff } |переведено специально для amovies . org сэм : йоу . '
                '|the long way home |♪ come here and be with me ♪ |( elevator bell ringing )'
                ' |_ bar _| синхронизация от elderman|sync , corrected by @ elder _ man '
                'переведено на cotranslate . net |== sync , corrected by elderman '
                '== @ elder _ man ',
                '', string)
            str2 = re.sub(
                '\*|~|†|•|¶|♪|#|\+|=|_|\. \. \. \. \.|\. \. \. \.|\. \. \.|\. \.',
                '', str1)
            str3 = re.sub(r'\s+', ' ', str2)
            str4 = re.sub('€', 'я', str3)
            str5 = re.sub('¬', 'в', str4)
            str6 = re.sub('\' \'', '"', str5)
            str7 = re.sub('ьı', 'ы', str6)
            str8 = re.sub('ƒ', 'д', str7)
            str9 = re.sub('є', 'е', str8)
            str10 = re.sub('ј', 'а', str9)
            str11 = re.sub('ў', 'у', str10)
            str12 = re.sub('∆', 'ж', str11)
            str13 = re.sub('√', 'г', str12)
            str14 = re.sub('≈', 'е', str13)
            str15 = re.sub('- - - - - -', '–', str14)
            str16 = re.sub('- - -', '–', str15)
            str17 = re.sub('- -', '–', str16)
            str18 = str17.strip('`').strip()
            str19 = str18.strip('–').strip()
            return str19

        def m(string):
            str1 = re.sub('mне', 'мне', string)
            str2 = re.sub('mама', 'мама', str1)
            str3 = re.sub('mистер', 'мистер', str2)
            str4 = re.sub('mеня', 'меня', str3)
            str5 = re.sub('mы', 'мы', str4)
            str6 = re.sub('mэлоун', 'мэлоун', str5)
            str7 = re.sub('mой', 'мой', str6)
            str8 = re.sub('mисс', 'мисс', str7)
            str9 = re.sub('mоя', 'моя', str8)
            str10 = re.sub('mакфарленд', 'макфарленд', str9)
            str11 = re.sub('m - м - м', 'м - м - м', str10)
            str12 = re.sub('mм', 'мм', str11)
            str13 = re.sub('mарек', 'марек', str12)
            str14 = re.sub('maмa', 'мaмa', str13)
            str15 = re.sub('mнe', 'мнe', str14)
            str16 = re.sub('moй', 'мoй', str15)
            str17 = re.sub('mapeк', 'мapeк', str16)
            return str17

        def s(string):
            str1 = re.sub('ѕодожди', 'подожди', string)
            str2 = re.sub('ѕока', 'пока', str1)
            str3 = re.sub('ѕривет', 'привет', str2)
            str4 = re.sub('ѕорис', 'борис', str3)
            str5 = re.sub('ѕоже', 'боже', str4)
            str6 = re.sub('ѕожалуйста', 'пожалуйста', str5)
            str7 = re.sub('ѕилл', 'билл', str6)
            str8 = re.sub('ѕерни', 'берни', str7)
            str9 = re.sub('ѕольше', 'больше', str8)
            str10 = re.sub('ѕочему', 'почему', str9)
            str11 = re.sub('ѕрости', 'прости', str10)
            str12 = re.sub('ѕольшое', 'большое', str11)
            str13 = re.sub('ѕрайн', 'брайн', str12)
            str14 = re.sub('ѕапа', 'папа', str13)
            str15 = re.sub('ѕонимаешь', 'понимаешь', str14)
            str16 = re.sub('ѕлагодарю', 'благодарю', str15)
            str17 = re.sub('ѕогоди', 'погоди', str16)
            str18 = re.sub('ѕо', 'по', str17)
            str19 = re.sub('ѕойдем', 'пойдем', str18)
            str20 = re.sub('ѕэм', 'пэм', str19)
            str21 = re.sub('ѕотому', 'потому', str20)
            str22 = re.sub('ѕонни', 'бонни', str21)
            str23 = re.sub('ѕарень', 'парень', str22)
            str24 = re.sub('ѕалермо', 'палермо', str23)
            str25 = re.sub('ѕросай', 'бросай', str24)
            str26 = re.sub('ѕеги', 'беги', str25)
            str27 = re.sub('ѕошли', 'пошли', str26)
            str28 = re.sub('ѕослушайте', 'послушайте', str27)
            str29 = re.sub('ѕриятно', 'приятно', str28)
            return str29

        def n(string):
            str1 = re.sub('ќтойди', 'отойди', string)
            str2 = re.sub('ќна', 'она', str1)
            str3 = re.sub('ќ !', 'о !', str2)
            str4 = re.sub('ќ ,', 'о ,', str3)
            str5 = re.sub('ќпять', 'опять', str4)
            str6 = re.sub('ќк', 'ок', str5)
            str7 = re.sub('ќни', 'они', str6)
            str8 = re.sub('ќуе', 'оуе', str7)
            str9 = re.sub('ќн', 'он', str8)
            str10 = re.sub('ќ', 'н', str9)
            return str10

        def jo(string):
            str1 = re.sub('ёто', 'это', string)
            str2 = re.sub('ёдди', 'эдди', str1)
            str3 = re.sub('ённи', 'энни', str2)
            str4 = re.sub('ёйприл', 'эйприл', str3)
            return str4

        def cav(string):
            str1 = re.sub('" ак', 'так', string)
            str2 = re.sub('" звини', 'извини', str1)
            str3 = re.sub('" ерт !', 'берт !', str2)
            str4 = re.sub('" еперь', 'теперь', str3)
            str5 = re.sub('" о', 'то', str4)
            str6 = re.sub('сегодня "', 'сегодня ?', str5)
            str7 = re.sub('днем "', 'днем ', str6)
            str8 = re.sub('" олько', 'только', str7)
            str9 = re.sub('" ы', 'ты', str8)
            str10 = re.sub('" йди', 'уйди', str9)
            str11 = re.sub('ц не', 'не', str10)
            str12 = re.sub('ц ѕонимаешь', 'понимаешь', str11)
            str13 = re.sub('ц  пока', 'пока', str12)
            str14 = re.sub('" аппи', 'паппи', str13)
            str15 = re.sub('ц " ,', '', str14)
            return str15

        def h(string):
            str1 = re.sub('ћадно', 'ладно', string)
            str2 = re.sub('ћюди', 'люди', str1)
            str3 = re.sub('ћ', 'м', str2)
            return str3

        def ch(string):
            str1 = re.sub('ч да', 'да', string)
            str2 = re.sub('ч ћожет', 'может', str1)
            str3 = re.sub('ч ну', 'ну', str2)
            str4 = re.sub('ч ага', 'ага', str3)
            return str4

        clean_text = ch(h(cav(jo(n(s(m(clean(text))))))))

        return clean_text




