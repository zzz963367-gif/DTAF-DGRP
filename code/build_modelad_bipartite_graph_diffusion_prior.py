import numpy as np
import pandas as pd

def pair_frame(df):
    value_col = "IC50" if "IC50" in df.columns else df.columns[-1]
    return pd.DataFrame(
        {
            "Cell": df.iloc[:, 0].astype(str),
            "Drug": df.iloc[:, 1].astype(str),
            "IC50": df[value_col].astype(float),
        }
    )

def build_maps(frames):
    cells = sorted(set().union(*[set(df["Cell"]) for df in frames.values()]))
    drugs = sorted(set().union(*[set(df["Drug"]) for df in frames.values()]))
    return {cell: idx for idx, cell in enumerate(cells)}, {drug: idx for idx, drug in enumerate(drugs)}

def build_matrix(frames, cell_to_id, drug_to_id):
    num_cells = len(cell_to_id)
    num_drugs = len(drug_to_id)
    values = np.zeros((num_cells, num_drugs), dtype=np.float32)
    mask = np.zeros((num_cells, num_drugs), dtype=bool)
    for _, row in frames["train"].iterrows():
        c = cell_to_id[str(row["Cell"])]
        d = drug_to_id[str(row["Drug"])]
        values[c, d] = float(row["IC50"])
        mask[c, d] = True

    global_mean = float(frames["train"]["IC50"].mean())
    global_std = float(frames["train"]["IC50"].std())
    drug_mean = np.zeros(num_drugs, dtype=np.float32)
    drug_std = np.ones(num_drugs, dtype=np.float32)
    for drug, drug_id in drug_to_id.items():
        observed = values[:, drug_id][mask[:, drug_id]]
        if observed.size == 0:
            drug_mean[drug_id] = global_mean
            drug_std[drug_id] = global_std
        else:
            drug_mean[drug_id] = float(observed.mean())
            std = float(observed.std())
            drug_std[drug_id] = std if std > 1e-6 else global_std

    norm_values = np.zeros_like(values)
    rows, cols = np.where(mask)
    norm_values[rows, cols] = (values[rows, cols] - drug_mean[cols]) / drug_std[cols]
    return values, norm_values, mask, drug_mean, drug_std

def split_indices(frame, cell_to_id, drug_to_id):
    cells = np.array([cell_to_id[str(cell)] for cell in frame["Cell"]], dtype=np.int64)
    drugs = np.array([drug_to_id[str(drug)] for drug in frame["Drug"]], dtype=np.int64)
    y = frame["IC50"].to_numpy(np.float32)
    return cells, drugs, y

def l2_normalize(features):
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm[norm < 1e-8] = 1.0
    return features / norm

def knn_graph_from_features(features, k, temperature, self_weight):
    features = l2_normalize(features.astype(np.float32))
    sim = features @ features.T
    np.fill_diagonal(sim, -np.inf)
    k = min(k, sim.shape[1] - 1)
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    row = np.arange(sim.shape[0])[:, None]
    scores = sim[row, idx]
    scores = np.maximum(scores, -0.5)
    weights = np.exp(scores / max(temperature, 1e-6))
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    graph = np.zeros_like(sim, dtype=np.float32)
    graph[row, idx] = weights.astype(np.float32)
    if self_weight > 0.0:
        graph = graph + np.eye(graph.shape[0], dtype=np.float32) * self_weight
    graph = graph / np.maximum(graph.sum(axis=1, keepdims=True), 1e-8)
    return graph.astype(np.float32)

def response_profile_features(norm_values, mask, axis):
    observed = norm_values * mask.astype(np.float32)
    counts = mask.sum(axis=axis, keepdims=True).astype(np.float32)
    counts[counts < 1.0] = 1.0
    if axis == 1:
        row_mean = observed.sum(axis=1, keepdims=True) / counts
        centered = (observed - row_mean) * mask.astype(np.float32)
    else:
        col_mean = observed.sum(axis=0, keepdims=True) / counts
        centered = ((observed - col_mean) * mask.astype(np.float32)).T
    return l2_normalize(centered.astype(np.float32))

def mix_graphs(response_graph, side_graph, response_weight):
    graph = response_weight * response_graph + (1.0 - response_weight) * side_graph
    graph = np.maximum(graph, 0.0)
    graph = graph / np.maximum(graph.sum(axis=1, keepdims=True), 1e-8)
    return graph.astype(np.float32)
