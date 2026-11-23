import numpy as np

def fit_calibrator(train_df, drug_lambda, cell_lambda, iterations):
    x = train_df["pred"].to_numpy(float)
    y = train_df["real"].to_numpy(float)
    design = np.vstack([x, np.ones_like(x)]).T
    global_coef = np.linalg.solve(design.T @ design + 1e-6 * np.eye(2), design.T @ y)

    drug_groups = list(train_df.groupby("Drug"))
    cell_groups = list(train_df.groupby("Cell"))
    drug_coef = {}
    for drug, group in drug_groups:
        xd = group["pred"].to_numpy(float)
        yd = group["real"].to_numpy(float)
        design_d = np.vstack([xd, np.ones_like(xd)]).T
        drug_coef[drug] = np.linalg.solve(
            design_d.T @ design_d + drug_lambda * np.eye(2),
            design_d.T @ yd + drug_lambda * global_coef,
        )

    cell_bias = {cell: 0.0 for cell, _ in cell_groups}
    drug_bias = {drug: 0.0 for drug, _ in drug_groups}
    for _ in range(iterations):
        for cell, group in cell_groups:
            base = np.array(
                [
                    drug_coef[drug][0] * pred + drug_coef[drug][1] + drug_bias[drug]
                    for drug, pred in zip(group["Drug"], group["pred"])
                ],
                dtype=float,
            )
            residual = group["real"].to_numpy(float) - base
            cell_bias[cell] = residual.sum() / (len(group) + cell_lambda)
        for drug, group in drug_groups:
            base = np.array(
                [
                    drug_coef[drug][0] * pred + drug_coef[drug][1] + cell_bias[cell]
                    for cell, pred in zip(group["Cell"], group["pred"])
                ],
                dtype=float,
            )
            residual = group["real"].to_numpy(float) - base
            drug_bias[drug] = residual.sum() / (len(group) + drug_lambda)

    return {
        "global_coef": global_coef,
        "drug_coef": drug_coef,
        "cell_bias": cell_bias,
        "drug_bias": drug_bias,
        "drug_lambda": drug_lambda,
        "cell_lambda": cell_lambda,
        "iterations": iterations,
    }

def apply_calibrator(df, calibrator):
    global_coef = calibrator["global_coef"]
    drug_coef = calibrator["drug_coef"]
    cell_bias = calibrator["cell_bias"]
    drug_bias = calibrator["drug_bias"]
    output = []
    for cell, drug, pred in zip(df["Cell"], df["Drug"], df["pred"]):
        coef = drug_coef.get(drug, global_coef)
        output.append(coef[0] * float(pred) + coef[1] + cell_bias.get(cell, 0.0) + drug_bias.get(drug, 0.0))
    return np.asarray(output, dtype=np.float32)
