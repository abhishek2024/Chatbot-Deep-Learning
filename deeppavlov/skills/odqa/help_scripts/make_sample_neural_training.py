from deeppavlov.core.common.file import read_json, save_json

TRAIN_PATH = '/media/olga/Data/datasets/squad/preproc/train-v1.1_negative_samples.json'
DEV_PATH = '/media/olga/Data/datasets/squad/preproc/dev-v1.1_negative_samples.json'

tr = read_json(TRAIN_PATH)
dev = read_json(DEV_PATH)

tr_sample = tr[:100]
dev_sample = dev[:10]

save_json(tr_sample, '/media/olga/Data/datasets/squad/preproc/train-sample--v1.1_negative_samples.json')
save_json(dev_sample, '/media/olga/Data/datasets/squad/preproc/dev-sample--v1.1_negative_samples.json')

