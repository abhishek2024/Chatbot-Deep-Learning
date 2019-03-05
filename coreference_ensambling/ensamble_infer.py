from pathlib import Path

from coreference_ensambling.ensembling import get_ensemble_prediction
from coreference_ensambling.infer import prepare_ensemble, write_predictions
# from coreference_ensambling.infer import get_model_docs_prediction

model_config = Path("/home/mks/projects/DeepPavlov/deeppavlov/configs/coreference_resolution/test_ensamble/ancor_rucor_full/model_bert_mid.json")

test_path = Path('/home/mks/.deeppavlov/downloads/AnCor/test_gold.jsonl')
test_lm_path = '/home/mks/.deeppavlov/downloads/embeddings/test_gold_bert_mid.hdf5'

test_data = Path("/home/mks/Work/Datasets/AnCor/ForTest_NoAnswers+morph/")
mentions_path = test_data.joinpath("Mentions")
morph_files = test_data.joinpath("morph")
prediction_root = Path("/home/mks/Downloads/tmp/")

# config = prepare_ensemble(model_config, test_path, test_lm_path, ensemble=False)
# pred_dict = get_model_docs_prediction(config, mode='test')
# write_predictions(pred_dict['predicted_clusters'], prediction_root, morph_files)

ensemble_configs = prepare_ensemble(model_config, test_path, test_lm_path, ensemble=True, cv_tmp=False)
ensemble_predicted_clusters, mention_to_predicted = get_ensemble_prediction([ensemble_configs[0]], mode='test')
write_predictions(ensemble_predicted_clusters, prediction_root, morph_files)
