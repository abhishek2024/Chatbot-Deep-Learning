from pathlib import Path

from coreference_ensambling.ensembling import get_ensemble_prediction
from coreference_ensambling.infer import prepare_ensemble, write_predictions
# from coreference_ensambling.infer import get_model_docs_prediction

# model_config = Path("/home/mks/projects/DeepPavlov/deeppavlov/configs/coreference_resolution/test_ensamble/ancor_rucor_full/model_bert_mid.json")

ensemble_root = Path("/home/mks/projects/DeepPavlov/deeppavlov/configs/coreference_resolution/test_ensamble/comp_final/full/ancor_rucor_folds")

# test_path = Path('/home/yurakuratov/.deeppavlov/downloads/ancor/ancor_final/test_final.jsonl')
# test_lm_path = '/home/yurakuratov/.deeppavlov/downloads/embeddings/test_final_bert_mid.hdf5'

test_data = Path("/home/mks/Work/Datasets/AnCor/ForTest_NoAnswers+morph/")
mentions_path = test_data.joinpath("Mentions")
morph_files = test_data.joinpath("morph")
prediction_root = Path("/home/mks/Download/tmp/final_predictions_full")

# config = prepare_ensemble(model_config, test_path, test_lm_path, ensemble=False)
# pred_dict = get_model_docs_prediction(config, mode='test')
# write_predictions(pred_dict['predicted_clusters'], prediction_root, morph_files)

ensemble_configs = [p for p in ensemble_root.glob("*.json")]

# ensemble_configs = prepare_ensemble(model_config, test_path, test_lm_path, ensemble=True, cv_tmp=False)
ensemble_predicted_clusters, mention_to_predicted = get_ensemble_prediction(ensemble_configs, mode='test')

print('\nsolo clusters removed')
for d in ensemble_predicted_clusters:
    ensemble_predicted_clusters[d] = [list(set(c)) for c in ensemble_predicted_clusters[d] if len(set(c)) > 1]

for d in mention_to_predicted:
    for m in list(mention_to_predicted[d].keys()):
        if len(set(mention_to_predicted[d][m])) == 1:
            del mention_to_predicted[d][m]


write_predictions(ensemble_predicted_clusters, prediction_root, morph_files)
