import os
import json

data = []

for root, _, files in os.walk("/media/olga/Data/Downloads/ai-2634736723902537328"):
    for fname in files:
        with open(os.path.join(root, fname)) as fin:
            content = fin.read()
        data.append({"text": content, "title": fname})

data = json.dumps(data)

with open("ai-2634736723902537328.txt", 'w') as fout:
    for item in data:
        fout.write(item)
        fout.write('\n')
