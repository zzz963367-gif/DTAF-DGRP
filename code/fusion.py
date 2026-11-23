import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,roc_auc_score,f1_score,recall_score

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    threshold = np.median(y_true)
    y_bin = (y_true < threshold).astype(int)
    pred_bin = (y_pred < threshold).astype(int)
    return {
        "PCC": pearsonr(y_true, y_pred)[0],
        "SCC": spearmanr(y_true, y_pred)[0],
        "MSE": mse,
        "RMSE": mse**0.5,
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "AUC": roc_auc_score(y_bin, -y_pred),
        "F1": f1_score(y_bin, pred_bin),
        "Recall": recall_score(y_bin, pred_bin),
    }

def oof_predictions(x, y, alpha, cv_folds, cv_seed):
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=cv_seed)
    pred = np.empty(len(y), dtype=float)
    for train_idx, heldout_idx in splitter.split(x):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_heldout = scaler.transform(x[heldout_idx])
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(x_train, y[train_idx])
        pred[heldout_idx] = model.predict(x_heldout)
    return pred
