from datetime import datetime
import pandas as pd

from src.config import RAW_DATA_PATH, DEA_PROCESSED_PATH
from src.dea.compute_dea import compute_two_stage_ddf_ndea


def main():
    run_start_dt = datetime.now()
    print(f"[RUN] Started at {run_start_dt:%Y-%m-%d %H:%M:%S}")

    # ==========================================================
    # Load raw data
    # ==========================================================
    data = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded raw data from: {RAW_DATA_PATH}")
    print(f"Number of rows: {len(data)}")

    # ==========================================================
    # Run DEA
    # ==========================================================
    dea_results = compute_two_stage_ddf_ndea(
        data,
        group_col="Year" if "Year" in data.columns else None,
        dmu_col="DMU" if "DMU" in data.columns else None,
        w1=0.5,
        w2=0.5,
        solver_msg=False
    )

    print("\nDEA results preview:")
    print(dea_results.head())

    # ==========================================================
    # Merge DEA results back into original data
    # ==========================================================
    merge_cols = []

    if "DMU" in data.columns and "DMU" in dea_results.columns:
        merge_cols.append("DMU")

    if "Year" in data.columns and "Year" in dea_results.columns:
        merge_cols.append("Year")

    if merge_cols:
        final_data = data.merge(dea_results, on=merge_cols, how="left")
    else:
        final_data = pd.concat(
            [data.reset_index(drop=True), dea_results.reset_index(drop=True)],
            axis=1
        )

    # ==========================================================
    # Save processed DEA output for ML
    # ==========================================================
    final_data.to_csv(DEA_PROCESSED_PATH, index=False)
    print(f"\nSaved DEA processed dataset to: {DEA_PROCESSED_PATH}")

    run_end_dt = datetime.now()
    elapsed = run_end_dt - run_start_dt

    print(f"\n[RUN] Ended at   {run_end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"[RUN] Elapsed    {elapsed}")
    print("\n====== Two-Stage DDF Network DEA Analysis Complete ======")


if __name__ == "__main__":
    main()