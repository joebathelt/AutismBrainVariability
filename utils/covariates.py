"""
Covariate residualisation shared across the fMRI analysis scripts
(C1, C3, C3b). Mirrors the regression used in C1 so landscape analyses
inherit the same covariate-adjustment semantics.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


COVARIATES = ("Age_in_Yrs", "Gender", "FS_IntraCranial_Vol",
              "Movement_RelativeRMS_mean")

ANCESTRY_PC_PREFIX = "ancestry_PC"


def ancestry_pc_columns(n_pcs):
    """Names of the ancestry PC covariate columns (ancestry_PC1..ancestry_PC{n})."""
    return [f"{ANCESTRY_PC_PREFIX}{i}" for i in range(1, n_pcs + 1)]


def load_ancestry_pcs(eigenvec_path, n_pcs):
    """Load the leading n_pcs PCs from a plink2 --pca .eigenvec file.

    Parameters
    ----------
    eigenvec_path : str or Path
        Path to a plink2 .eigenvec output. Header line begins with '#FID'
        (or '#IID' for plink2 builds that drop FID), followed by PC1, PC2…
    n_pcs : int
        Number of leading PCs to retain.

    Returns
    -------
    pd.DataFrame
        Indexed by Subject (str — the IID). Columns ancestry_PC1..ancestry_PC{n_pcs}.
    """
    df = pd.read_csv(eigenvec_path, sep=r"\s+", engine="python")
    df.columns = [c.lstrip("#") for c in df.columns]
    if "IID" not in df.columns:
        raise RuntimeError(
            f"{eigenvec_path}: expected an IID column; got {list(df.columns)}"
        )
    pc_src = [f"PC{i}" for i in range(1, n_pcs + 1)]
    missing = [c for c in pc_src if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{eigenvec_path}: missing PC columns {missing}; "
            f"available: {list(df.columns)}"
        )
    out = df.set_index(df["IID"].astype(str))[pc_src].copy()
    out.columns = ancestry_pc_columns(n_pcs)
    out.index.name = "Subject"
    return out


def regress_out_covariates(Y, covariate_df):
    """Return residuals of Y after regressing on covariate_df via OLS.

    Parameters
    ----------
    Y : pd.DataFrame or pd.Series
        Target variable(s). Index and column names are preserved on output.
    covariate_df : pd.DataFrame
        Covariates for the same samples as Y (same index). Categorical
        columns (e.g. Gender) are dummy-encoded with drop_first=True,
        matching C1_run_univariate_fMRI_prediction.py.

    Returns
    -------
    residuals : pd.DataFrame or pd.Series
        Same type, index, and columns as Y.
    """
    X = pd.get_dummies(covariate_df, drop_first=True).astype(float).values

    if isinstance(Y, pd.Series):
        y_arr = Y.values.astype(float)
        model = LinearRegression().fit(X, y_arr)
        resid = y_arr - model.predict(X)
        return pd.Series(resid, index=Y.index, name=Y.name)

    Y_arr = Y.values.astype(float)
    resid = np.empty_like(Y_arr)
    for i in range(Y_arr.shape[1]):
        model = LinearRegression().fit(X, Y_arr[:, i])
        resid[:, i] = Y_arr[:, i] - model.predict(X)
    return pd.DataFrame(resid, index=Y.index, columns=Y.columns)
