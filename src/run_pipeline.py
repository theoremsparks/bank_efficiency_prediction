from src.run_dea import main as run_dea_main
from src.run_analysis import main as run_analysis_main


def main():
    print("=" * 70)
    print("STARTING FULL DEA + ANALYSIS PIPELINE")
    print("=" * 70)

    print("\nStep 1: Running DEA...")
    run_dea_main()

    print("\nStep 2: Running analysis...")
    run_analysis_main()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()