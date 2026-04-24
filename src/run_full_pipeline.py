from src.run_dea import main as run_dea_main
from src.run_analysis import main as run_analysis_main
from src.ml.train_voting_optimized_classifier import main as run_voting_main


def main():
    print("=" * 80)
    print("STARTING FULL END-TO-END PIPELINE")
    print("=" * 80)

    print("\nStep 1: DEA")
    run_dea_main()

    print("\nStep 2: Analysis")
    run_analysis_main()

    print("\nStep 3: Final Machine Learning Model (Voting Classifier)")
    run_voting_main()

    print("\n" + "=" * 80)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nSelected model: Voting Classifier")


if __name__ == "__main__":
    main()