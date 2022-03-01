"""
    1. For X: Create an array of 2d arrays, n samples(trials) * cell heatmap (a heatmap is a 2d array of timepoints * n cells)
    2. For y: create a 1d array: the outcome is each element

For one mouse:

d = {
    Test 1 (Block 1) : [(X, [large csvs, small csvs, omission csvs]), (y, ["large", "small", and "omission" labels- same len as the x list])]
}

"""
import os, glob
from tarfile import BLOCKSIZE
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def find_paths(root_path: Path, block, mouse, session, startswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, block, "**", mouse, session, f"{startswith}*"),
        recursive=True,
    )
    return files


def dissect_outcome(path: Path):
    return list(path.parts)[8]


# /media/rory/Padlock_DT/BLA_Analysis/Decoding/Arranged_Dataset/{1.0}/[Large]/{BLA-Insc-1}/{RDT D1}/[trail_1.csv]
def binary_logreg():
    ROOT_PATH = Path(r"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Arranged_Dataset/")
    block = "1.0"
    session = "RDT D1"
    mouse = "BLA-Insc-1"

    files = find_paths(ROOT_PATH, block, mouse, session, "trail")

    # print(*files, sep="\n")
    print("Number of trials (csvs): ", len(files))

    y = []
    X = []

    for csv in files:
        csv = Path(csv)
        outcome = dissect_outcome(csv)
        df: pd.DataFrame
        df = pd.read_csv(csv)
        df = df.T
        df = df.iloc[1:, :]  # remove first row (the cell names)
        # go through columns and add to X and y
        for col in list(df.columns):
            X.append(list(df[col]))
            y.append(outcome)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, train_size=0.80, random_state=0
    )

    print("Number of samples: ", len(X_train))
    # print(f"y : {y}")

    # l2 -> all neurons make small contributions
    # l1 -> only some neurons make large contributions
    model_pipe = make_pipeline(
        StandardScaler(), LogisticRegression(penalty="l2", max_iter=200)
    )

    # to record scores across multiple classifiers
    model_results = {}
    f1_scores = cross_val_score(model_pipe, X_train, y_train, scoring="f1_macro")
    avg_f1 = np.average(f1_scores)
    print("Average F1 Score: ", avg_f1)
    # print(type(f1_scores))

    model_pipe.fit(X_train, y_train)
    print("Model accuracy: %.3f" % model_pipe.score(X_test, y_test))


"""def load_data() -> Tuple[np.ndarray, np.ndarray]:

    ...


from sklearn.decomposition import PCA


def create_model(model):
    return make_pipeline(StandardScaler(), PCA(3), model)


X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y)

reslts = {}

logistic_pipe = create_model(LogisticRegression())
svm_pipe = create_model(SVC(C=1000))

# RandomForestClassifier(n_estimators=)
results["logistc"] = cross_val_score(logistic_pipe, X, y, scor)
svm

# model = make_pipeline(StandardScaler(), LogisticRegression())"""


def multinomial_logreg():
    ROOT_PATH = Path(r"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Arranged_Dataset/")
    session = "RDT D1"
    mouse = "BLA-Insc-1"
    blocks = ["1.0", "2.0", "3.0"]
    for block in blocks:
        print()
        print(f"PREDICTING OUTCOME IN BLOCK {block}, {mouse} {session}")
        files = find_paths(ROOT_PATH, block, mouse, session, "trail")
        files_small_large = [i for i in files if not "Omission" in i and not "ITI" in i]
        # print(*files, sep="\n")
        print("Number of trials (csvs): ", len(files_small_large))

        y = []
        X = []

        for csv in files_small_large:
            csv = Path(csv)
            outcome = dissect_outcome(csv)
            df: pd.DataFrame
            df = pd.read_csv(csv)
            df = df.T
            df = df.iloc[1:, :]  # remove first row (the cell names)
            # go through columns and add to X and y
            for col in list(df.columns):
                X.append(list(df[col]))
                y.append(outcome)

        # pull only unique elements in list
        labels = []
        for ele in y:
            if ele not in labels:
                labels.append(ele)
        print(f"Labels: {labels}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, train_size=0.80, random_state=0
        )

        print("Number of samples: ", len(X_train))
        # print(f"y : {y}")

        # l2 -> all neurons make small contributions
        # l1 -> only some neurons make large contributions
        model_pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty="l2", max_iter=400, multi_class="multinomial"),
        )

        # to record scores across multiple classifiers
        model_results = {}
        f1_scores = cross_val_score(model_pipe, X_train, y_train, scoring="f1_macro")
        avg_f1 = np.average(f1_scores)
        print("Average F1 Score: ", avg_f1)
        # print(type(f1_scores))

        model_pipe.fit(X_train, y_train)
        print("Model accuracy: %.3f" % model_pipe.score(X_test, y_test))


if __name__ == "__main__":
    multinomial_logreg()
