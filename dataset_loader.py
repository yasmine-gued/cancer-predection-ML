import pandas as pd
from sklearn.datasets import load_breast_cancer

import  pandas as pd

def load_breast_cancer_dataset():
    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="maladie")

    df = pd.concat([X, y], axis=1)

    # 0 = malignant => Oui (malade)
    # 1 = benign => Non (non malade)
    df["maladie"] = df["maladie"].map({0: "Oui", 1: "Non"})

    # ajouter un identifiant et un nom artificiel
    df.insert(0, "patient_id", range(1, len(df) + 1))
    df.insert(1, "nom", [f"Patiente_{i}" for i in range(1, len(df) + 1)])

    return df


if __name__ == "__main__":
    df = load_breast_cancer_dataset()
    print(df.head())
    print("Nombre de patientes :", len(df))
    df.to_csv("breast_cancer_dataset.csv", index=False)
    print("Dataset sauvegardé dans breast_cancer_dataset.csv")