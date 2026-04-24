import numpy as np
import pandas as pd
import pulp


def compute_two_stage_ddf_ndea(
    df,
    group_col="Year",
    dmu_col="DMU",
    w1=0.5,
    w2=0.5,
    solver_msg=False
):
    """
    Compute a two-stage network DDF-DEA with:

      - Stage 1 controllable inputs: Assets, Employee Expense
      - Stage 1 quasi-fixed input: Equity
      - Stage 1 intermediate output: Deposits

      - Stage 2 controllable input: Borrowings
      - Stage 2 undesirable input: NPAs (Previous Period)
      - Stage 2 desirable outputs:
            Performing Loans,
            Investment,
            Net Income,
            Net-interest Income,
            Non-interest Income
      - Stage 2 undesirable output: NPAs

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing all DEA variables.
    group_col : str or None, default="Year"
        Column used to build separate frontiers, for example yearly frontiers.
        If None, one common frontier is used for all DMUs.
    dmu_col : str or None, default="DMU"
        Column identifying each DMU.
    w1 : float, default=0.5
        Weight for stage 1 inefficiency.
    w2 : float, default=0.5
        Weight for stage 2 inefficiency.
    solver_msg : bool, default=False
        Whether the PuLP CBC solver should print messages.

    Returns
    -------
    pandas.DataFrame
        DEA results for each DMU.
    """

    # ==========================================================
    # Variable definitions
    # ==========================================================
    stage1_x_cols = ["Assets", "Employee Expense"]      # controllable inputs stage 1
    stage1_f_cols = ["Equity"]                          # quasi-fixed input stage 1
    z_cols = ["Deposits"]                               # intermediate output / input

    stage2_q_cols = ["Borrowings"]                      # controllable input stage 2
    stage2_b_cols = ["NPAs (Previous Period)"]          # undesirable input stage 2

    good_y_cols = [
        "Performing Loans",
        "Investment",
        "Net Income",
        "Net-interest Income",
        "Non-interest Income"
    ]                                                   # desirable outputs stage 2

    bad_u_cols = ["NPAs"]                               # undesirable output stage 2

    # ==========================================================
    # Validate required columns
    # ==========================================================
    required_cols = (
        stage1_x_cols
        + stage1_f_cols
        + z_cols
        + stage2_q_cols
        + stage2_b_cols
        + good_y_cols
        + bad_u_cols
    )

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in DEA input data: {missing_cols}")

    if dmu_col is not None and dmu_col not in df.columns:
        raise ValueError(f"dmu_col '{dmu_col}' not found in dataframe columns.")

    if group_col is not None and group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not found in dataframe columns.")

    # ==========================================================
    # Copy data and scale large-value columns for stability
    # ==========================================================
    df = df.copy()

    scale_cols = (
        stage1_x_cols
        + stage1_f_cols
        + z_cols
        + stage2_q_cols
        + stage2_b_cols
        + good_y_cols
        + bad_u_cols
    )

    df_scaled = df.copy()

    for c in scale_cols:
        max_abs = df_scaled[c].abs().max()
        if pd.notna(max_abs) and max_abs > 1e5:
            df_scaled[c] = df_scaled[c] / 1_000_000.0

    # ==========================================================
    # Build groups
    # ==========================================================
    if group_col is None:
        groups = [("__ALL__", df_scaled.index.tolist())]
    else:
        groups = [(g, gdf.index.tolist()) for g, gdf in df_scaled.groupby(group_col)]

    results = []

    # ==========================================================
    # Solve one frontier group at a time
    # ==========================================================
    for group_name, idx_list in groups:
        sub = df_scaled.loc[idx_list].copy().reset_index()
        n = len(sub)

        if n == 0:
            continue

        # Arrays
        X = sub[stage1_x_cols].values
        F = sub[stage1_f_cols].values
        Z = sub[z_cols].values
        Q = sub[stage2_q_cols].values
        B = sub[stage2_b_cols].values
        Y = sub[good_y_cols].values
        U = sub[bad_u_cols].values

        # Positive / negative split for desirable outputs only
        Y_pos = np.clip(Y, 0, None)
        Y_neg = np.clip(-Y, 0, None)

        I = X.shape[1]
        K = F.shape[1]
        D = Z.shape[1]
        L = Q.shape[1]
        P = B.shape[1]
        R = Y.shape[1]
        S = U.shape[1]

        print(f"[INFO] Solving frontier for group = {group_name} with {n} DMUs")

        # ==========================================================
        # Solve the integrated two-stage model for each DMU
        # ==========================================================
        for o in range(n):
            prob = pulp.LpProblem(
                f"TwoStage_Network_DDF_{group_name}_{o+1}",
                pulp.LpMaximize
            )

            # Inefficiency variables
            beta1 = pulp.LpVariable("beta1", lowBound=0)
            beta2 = pulp.LpVariable("beta2", lowBound=0)

            # Intensity variables
            lam = [pulp.LpVariable(f"lam_{j}", lowBound=0) for j in range(n)]  # stage 1
            mu = [pulp.LpVariable(f"mu_{j}", lowBound=0) for j in range(n)]    # stage 2

            # Objective
            prob += w1 * beta1 + w2 * beta2, "Objective_Overall_Inefficiency"

            # ======================================================
            # Stage 1 constraints
            # ======================================================

            # controllable inputs
            for i in range(I):
                prob += (
                    pulp.lpSum(lam[j] * X[j, i] for j in range(n))
                    <= (1 - beta1) * X[o, i]
                ), f"Stage1_X_{i}"

            # quasi-fixed inputs
            for k in range(K):
                prob += (
                    pulp.lpSum(lam[j] * F[j, k] for j in range(n))
                    <= F[o, k]
                ), f"Stage1_F_{k}"

            # intermediate outputs
            for d in range(D):
                prob += (
                    pulp.lpSum(lam[j] * Z[j, d] for j in range(n))
                    >= Z[o, d]
                ), f"Stage1_Z_{d}"

            # ======================================================
            # Stage 2 constraints
            # ======================================================

            # intermediate inputs
            for d in range(D):
                prob += (
                    pulp.lpSum(mu[j] * Z[j, d] for j in range(n))
                    <= Z[o, d]
                ), f"Stage2_Z_{d}"

            # controllable inputs
            for l in range(L):
                prob += (
                    pulp.lpSum(mu[j] * Q[j, l] for j in range(n))
                    <= (1 - beta2) * Q[o, l]
                ), f"Stage2_Q_{l}"

            # undesirable inputs
            for p in range(P):
                prob += (
                    pulp.lpSum(mu[j] * B[j, p] for j in range(n))
                    == (1 - beta2) * B[o, p]
                ), f"Stage2_B_{p}"

            # desirable outputs with mixed-sign treatment
            for r in range(R):
                prob += (
                    pulp.lpSum(mu[j] * Y_pos[j, r] for j in range(n))
                    >= (1 + beta2) * Y_pos[o, r]
                ), f"Stage2_Ypos_{r}"

                prob += (
                    pulp.lpSum(mu[j] * Y_neg[j, r] for j in range(n))
                    <= (1 - beta2) * Y_neg[o, r]
                ), f"Stage2_Yneg_{r}"

            # undesirable outputs
            for s in range(S):
                prob += (
                    pulp.lpSum(mu[j] * U[j, s] for j in range(n))
                    == (1 - beta2) * U[o, s]
                ), f"Stage2_U_{s}"

            # ======================================================
            # VRS convexity constraints
            # ======================================================
            prob += pulp.lpSum(lam[j] for j in range(n)) == 1, "VRS_Constraint_Stage1"
            prob += pulp.lpSum(mu[j] for j in range(n)) == 1, "VRS_Constraint_Stage2"

            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=solver_msg))

            status = pulp.LpStatus[prob.status]
            beta1_val = pulp.value(beta1)
            beta2_val = pulp.value(beta2)

            if beta1_val is None or beta2_val is None or status != "Optimal":
                stage1_eff = np.nan
                stage2_eff = np.nan
                overall_beta = np.nan
                overall_eff = np.nan
            else:
                overall_beta = w1 * beta1_val + w2 * beta2_val
                stage1_eff = 1 - beta1_val
                stage2_eff = (1 - beta2_val) / (1 + beta2_val)
                overall_eff = 1 - overall_beta

            row = {
                "LP_Group": group_name,
                "LP_Status": status,
                "beta1": beta1_val,
                "beta2": beta2_val,
                "overall_beta": overall_beta,
                "Stage1_Efficiency": stage1_eff,
                "Stage2_Efficiency": stage2_eff,
                "Overall_Efficiency": overall_eff,
            }

            # Keep identifiers
            original_idx = sub.loc[o, "index"]

            if dmu_col is not None and dmu_col in df.columns:
                row[dmu_col] = df.loc[original_idx, dmu_col]

            if group_col is not None and group_col in df.columns:
                row[group_col] = df.loc[original_idx, group_col]

            results.append(row)

    return pd.DataFrame(results)