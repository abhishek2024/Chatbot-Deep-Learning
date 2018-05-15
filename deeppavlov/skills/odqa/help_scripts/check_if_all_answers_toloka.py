import csv

CSV_PATH = '/media/olga/Data/projects/ODQA/data/PMEF/qa_to_photo_toloka_revised.csv'
TXT_PATH = '/media/olga/Data/projects/ODQA/data/PMEF/original/drones.txt'


def read_csv(csv_path):
    output = []
    with open(csv_path) as fin:
        reader = csv.reader(fin)
        next(reader)
        for item in reader:
            output.append({'question': item[0], 'answer': item[1]})
    return output


def read_txt(txt_path):
    with open(txt_path) as fin:
        return fin.read()


dataset = read_csv(CSV_PATH)
text = read_txt(TXT_PATH).lower()

for item in dataset:
    answer = item['answer'].strip().lower()
    if text.find(answer) == -1:
        print(answer)

