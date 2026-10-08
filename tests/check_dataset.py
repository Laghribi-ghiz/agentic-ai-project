import pandas as pd


DATASET_PATH = "data/raw/Telco-Customer-Churn.csv"


def main():
    df = pd.read_csv(DATASET_PATH)

    print("\n=== TELCO DATASET ===")

    print("Shape:", df.shape)

    print("\nColumns:")
    for column in df.columns:
        print("-", column)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nTarget distribution:")
    print(df["Churn"].value_counts())


if __name__ == "__main__":
    main()