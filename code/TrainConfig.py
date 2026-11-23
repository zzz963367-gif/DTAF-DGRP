EPOCHS = 100
batch_size = 64
lr = 1e-05
seed = 14946934
embedding_dim = None
heads = 8
fc_layer_num = 3
fc_layer_dim = [256, 128, 1, 1, 1, 1]
dropout_rate = 0.25

import os

use_mrna_denoising = os.environ.get("MRNA_DENOISING_INPUT", "1") == "1"
experiment_override = os.environ.get("MODELADV_EXPERIMENT_NAME", "").strip()
if experiment_override:
    experiment_name = experiment_override
elif use_mrna_denoising:
    denoise_tag = os.environ.get("MRNA_DENOISING_TAG", "m04n005").strip()
    if denoise_tag:
        experiment_name = f'MRNADenoiseAtomGeneOnly_{denoise_tag}_fold0'
    else:
        experiment_name = 'MRNADenoiseAtomGeneOnly_fold0'
elif os.environ.get("SCGPT_LATENT_ABLATION", "0") == "1":
    experiment_name = 'MRNAGeneOnlyScGPTNoFoundation_fold0'
elif os.environ.get("USE_SCGPT_LATENT", "0") != "1":
    experiment_name = 'MRNAAtomGeneOnlyNoDenoise_fold0'
else:
    experiment_name = 'MRNAGeneOnlyScGPT_fold0'
save_path_log = f'./result/{experiment_name}.txt'
save_path_prediction = f'./result/{experiment_name}.csv'
save_path_model = f'./result/{experiment_name}.pkl'
