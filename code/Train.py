from paths import DATA
import copy
import os
import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Model import *
from Model_DAE import Encoder, Decoder
from Data import *
from TrainConfig import *

setup_seed(seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 25
USE_MRNA_DENOISING = os.environ.get("MRNA_DENOISING_INPUT", "1") == "1"
MRNA_MASK_RATE = float(os.environ.get("MRNA_DENOISING_MASK_RATE", "0.04"))
MRNA_NOISE_STD = float(os.environ.get("MRNA_DENOISING_NOISE_STD", "0.005"))

encoder, _ = joblib.load(str(DATA / "PreTrain" / "PreTrain.pkl"))
encoder = encoder.to(DEVICE)
model = Predictor(
    embedding_dim, heads, fc_layer_num, fc_layer_dim, dropout_rate,
).to(DEVICE)

resume_requested = os.environ.get("RESUME_TRAINING_CHECKPOINT", "0") == "1"
checkpoint_path = save_path_model.replace('.pkl', '_wo_Ensemble.pkl')
start_epoch = 0
resume_best_val_loss = float('inf')
if resume_requested and os.path.exists(checkpoint_path):
    loaded_epoch, loaded_model, loaded_encoder = joblib.load(checkpoint_path)
    model.load_state_dict(loaded_model.state_dict())
    encoder.load_state_dict(loaded_encoder.state_dict())
    start_epoch = int(loaded_epoch) + 1
    log_guess = save_path_model.replace('.pkl', '_FullModel_wo_Ensemble_LossLog.txt')
    if os.path.exists(log_guess):
        try:
            log_frame = pd.read_csv(log_guess, sep='\t')
            if 'Val_MSE' in log_frame.columns and len(log_frame) > 0:
                resume_best_val_loss = float(log_frame['Val_MSE'].min())
        except Exception:
            resume_best_val_loss = float('inf')
    print(f"Resuming from checkpoint epoch {loaded_epoch}; start_epoch={start_epoch}")

loss_func = nn.MSELoss()
optimizer = optim.AdamW([
    {'params': encoder.parameters(), 'lr': 0.5 * lr},
    {'params': model.parameters()},
], lr=lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
)

my_collate = CollateFn()
train_loader = DataLoader(MyDataSet(Gtrain), batch_size=batch_size, shuffle=True, collate_fn=my_collate)
train_eval_loader = DataLoader(MyDataSet(Gtrain), batch_size=batch_size, shuffle=False, collate_fn=my_collate)
val_loader = DataLoader(MyDataSet(Gval), batch_size=batch_size, shuffle=False, collate_fn=my_collate)
test_loader = DataLoader(MyDataSet(Gtest), batch_size=batch_size, shuffle=False, collate_fn=my_collate)

log_file_path = save_path_model.replace('.pkl', '_FullModel_wo_Ensemble_LossLog.txt')
if not (resume_requested and os.path.exists(log_file_path)):
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write('Epoch\tTrain_MSE\tVal_MSE\tLR\n')

best_val_loss = resume_best_val_loss
best_model_wts = copy.deepcopy(model.state_dict())
best_encoder_wts = copy.deepcopy(encoder.state_dict())
early_stop_counter = 0
print(f'Experiment: {experiment_name} (gene-only robust mRNA atom-cell branch)')
print(f'Start training, target epochs: {EPOCHS}, start epoch: {start_epoch}')


def corrupt_mrna_for_training(gene_raw):
    if not USE_MRNA_DENOISING:
        return gene_raw
    if MRNA_MASK_RATE <= 0 and MRNA_NOISE_STD <= 0:
        return gene_raw

    corrupted = gene_raw
    if MRNA_MASK_RATE > 0:
        keep = (torch.rand_like(gene_raw) > MRNA_MASK_RATE).float()
        sample_mean = gene_raw.mean(dim=1, keepdim=True)
        corrupted = keep * corrupted + (1.0 - keep) * sample_mean
    if MRNA_NOISE_STD > 0:
        feature_std = gene_raw.std(dim=1, keepdim=True).clamp_min(1e-6)
        corrupted = corrupted + torch.randn_like(corrupted) * (MRNA_NOISE_STD * feature_std)
    return corrupted


for epoch in range(start_epoch, EPOCHS):
    model.train()
    encoder.train()
    train_loss = 0.0
    for it, (pyg_batch, gene_ft) in enumerate(train_loader):
        pyg_batch = pyg_batch.to(DEVICE)
        gene_ft = gene_ft.to(DEVICE)
        train_gene_ft = corrupt_mrna_for_training(gene_ft)
        prediction = model(
            pyg_batch.x, pyg_batch, encoder(train_gene_ft), train_gene_ft,
        )
        loss = loss_func(prediction.view(-1), pyg_batch.ic50.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / (it + 1)

    model.eval()
    encoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for it, (pyg_batch, gene_ft) in enumerate(val_loader):
            pyg_batch = pyg_batch.to(DEVICE)
            gene_ft = gene_ft.to(DEVICE)
            prediction = model(pyg_batch.x, pyg_batch, encoder(gene_ft), gene_ft)
            val_loss += loss_func(prediction.view(-1), pyg_batch.ic50.view(-1)).item()
    avg_val_loss = val_loss / (it + 1)

    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[1]['lr']
    is_best = avg_val_loss < best_val_loss
    if is_best:
        best_val_loss = avg_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        best_encoder_wts = copy.deepcopy(encoder.state_dict())
        joblib.dump((epoch, model, encoder), save_path_model.replace('.pkl', '_wo_Ensemble.pkl'))
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    print(f'Epoch {epoch}: Train MSE {avg_train_loss:.6f} | Val MSE {avg_val_loss:.6f} | LR: {current_lr:.6f} {"Best!" if is_best else ""}')
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(f'{epoch}\t{avg_train_loss:.6f}\t{avg_val_loss:.6f}\t{current_lr:.6f}\n')
    if not is_best:
        print(f'   |-- EarlyStopping counter: {early_stop_counter} out of {PATIENCE}')
    if early_stop_counter >= PATIENCE:
        print(f'Early stopping after {PATIENCE} epochs without validation improvement.')
        break

print('Training complete. Loading the best validation checkpoint for testing.')
model.load_state_dict(best_model_wts)
encoder.load_state_dict(best_encoder_wts)
model.eval()
encoder.eval()


def predict_loader(loader):
    real_values, pred_values = [], []
    with torch.no_grad():
        for pyg_batch, gene_ft in loader:
            pyg_batch = pyg_batch.to(DEVICE)
            gene_ft = gene_ft.to(DEVICE)
            prediction = model(pyg_batch.x, pyg_batch, encoder(gene_ft), gene_ft)
            real_values += pyg_batch.ic50.cpu().tolist()
            pred_values += prediction.view(-1).cpu().tolist()
    return real_values, pred_values


def frame_from_csv(csv_path):
    frame = pd.read_csv(csv_path, header=0)
    return pd.DataFrame({
        'Cell': frame.iloc[:, 0].astype(str),
        'Drug': frame.iloc[:, 1].astype(str),
        'real': frame.iloc[:, -1].astype(float),
    })


def train_frame_from_total():
    if train_set_path:
        return frame_from_csv(train_set_path)
    frame = pd.read_csv(total_set_path, header=0)
    cells = frame.iloc[:, 0].astype(str).tolist()
    drugs = frame.iloc[:, 1].astype(str).tolist()
    ic50s = frame.iloc[:, -1].astype(float).tolist()
    cells, drugs, ic50s = LenientRemove(cells, drugs, ic50s, val_set_path, test_set_path)
    return pd.DataFrame({'Cell': list(cells), 'Drug': list(drugs), 'real': list(ic50s)})


split_specs = {
    'train': (train_eval_loader, train_frame_from_total()),
    'val': (val_loader, frame_from_csv(val_set_path)),
    'test': (test_loader, frame_from_csv(test_set_path)),
}
for split_name, (loader, frame) in split_specs.items():
    split_real, split_pre = predict_loader(loader)
    if len(frame) != len(split_pre):
        raise ValueError(f'{split_name}: metadata rows {len(frame)} != predictions {len(split_pre)}')
    frame = frame.copy()
    frame['real'] = [float(x[0]) if isinstance(x, list) else float(x) for x in split_real]
    frame['pred'] = split_pre
    frame.to_csv(save_path_prediction.replace('.csv', f'_{split_name}_predictions.csv'), index=False)

    if split_name == 'test':
        result_path = save_path_prediction.replace('.csv', '_wo_Ensemble.csv')
        pd.DataFrame({'real': split_real, 'best_predict': split_pre}).to_csv(result_path, index=False)
        print(f'Final test results saved to {result_path}')
