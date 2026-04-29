"""
Covariate residualisation shared across the fMRI analysis scripts
(C1, C3, C3b). Mirrors the regression used in C1 so landscape analyses
inherit the same covariate-adjustment semantics.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


COVARIATES = ("Age_in_Yrs", "FS_IntraCranial_Vol",
              "Movement_RelativeRMS_mean", "Gender")


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
