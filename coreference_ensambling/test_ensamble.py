from pathlib import Path

from coreference_ensambling.competition_metrics import compute_competition_metrics, models_metrics
from coreference_ensambling.dp_metrics import compute_dp_metric_for_ensemble, compute_dp_metric_for_models
from coreference_ensambling.new_ensembling import get_ensemble_prediction
from coreference_ensambling.ensambling import get_data
# Paths
ensemble_root = Path("/home/mks/projects/DeepPavlov/deeppavlov/configs/coreference_resolution/test_ensamble/")
ensemble_call_fold = ensemble_root.joinpath("new_kfold_ens")
ensemble_simple_fold = ensemble_root.joinpath("kfold_ens")

ancor_data = Path("/home/mks/Work/Datasets/AnCor/ForTraining+morph/")
chains_files = ancor_data.joinpath("Chains")
morph_files = ancor_data.joinpath("morph")
prediction_root = Path("/home/mks/Downloads/tmp/")

# ensembles configs
simple_ensemble_configs = [p for p in ensemble_simple_fold.glob("*.json")]
diff_ensemble_configs = [p for p in ensemble_call_fold.glob("*.json")]
assert len(simple_ensemble_configs) >= 1

# ensemble prediction (MEDIANA !!!!!!!!!!)
data, gold_preds = get_data(simple_ensemble_configs[0], "test")
# ensemble_prediction, mention_to_predicted = get_ensemble_prediction(ensemble_configs[0:3], data, True)
ensemble_prediction, mention_to_predicted = get_ensemble_prediction(diff_ensemble_configs, "test", True)
# ______________________________________________________________________________________________________________________
# compute ensemble competition metrics
compute_competition_metrics(ensemble_prediction, prediction_root, chains_files, morph_files)

# compute ensemble DP metrics
compute_dp_metric_for_ensemble(ensemble_prediction, mention_to_predicted, diff_ensemble_configs[0], mode='test')

# Alone Models
models_metrics(simple_ensemble_configs, prediction_root, morph_files, chains_files)
compute_dp_metric_for_models(simple_ensemble_configs, mode='test')
