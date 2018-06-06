from nltk.tokenize import sent_tokenize
import json

INPUT_PATH = '/media/olga/Data/projects/ranker_test/ranker_test/wiki_article.txt'
OUTPUT_PATH = '/media/olga/Data/projects/ranker_test/ranker_test/wiki_chunks_2ranker.json'
SQUAD_LIMIT = 500

with open(INPUT_PATH) as fin:
    data = fin.read()
    sentences = sent_tokenize(data)

# all_chunks = []
#
# _len = 0
# chunks = []
# for sentence in sentences:
#     _len += len(sentence.split())
#     if _len > SQUAD_LIMIT:
#         all_chunks.append(' '.join(chunks))
#         _len = 0
#         chunks = []
#     chunks.append(sentence)
#
# with open(OUTPUT_PATH, 'w') as fout:
#         # fout.write(' '.join(chunk))
#         # fout.write('\n')
#     json.dump(all_chunks, fout, ensure_ascii=False)
#
#
# print('Done!')


all_chunks = []

_len = 0
chunks = []
i = 0
for sentence in sentences:
    _len += len(sentence.split())
    if _len > SQUAD_LIMIT:
        d = {'text': ' '.join(chunks), 'title': i}
        all_chunks.append(d)
        _len = 0
        chunks = []
        i += 1
    chunks.append(sentence)

if chunks:
    d = {'text': ' '.join(chunks), 'title': i}
    all_chunks.append(d)

with open(OUTPUT_PATH, 'w') as fout:
    # fout.write(' '.join(chunk))
    # fout.write('\n')
    json.dump(all_chunks, fout, ensure_ascii=False)

print('Done!')