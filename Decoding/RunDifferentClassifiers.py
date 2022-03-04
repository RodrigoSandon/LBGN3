"""
    1. For X: Create an array of 2d arrays, n samples(trials) * cell heatmap (a heatmap is a 2d array of timepoints * n cells)
    2. For y: create a 1d array: the outcome is each element

For one mouse:

d = {
    Test 1 (Block 1) : [(X, [large csvs, small csvs, omission csvs]), (y, ["large", "small", and "omission" labels- same len as the x list])]
}

"""
import os, glob
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from scipy import stats

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)


def find_paths(root_path: Path, block, mouse, session, startswith: str) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, block, "**", mouse, session, f"{startswith}*"),
        recursive=True,
    )
    return files


def find_paths_shock(
    root_path: Path, shock_intensity, mouse, startswith: str
) -> List[str]:
    files = glob.glob(
        os.path.join(root_path, "**", shock_intensity, mouse, f"{startswith}*"),
        recursive=True,
    )
    return files


def dissect_outcome(path: Path):
    return list(path.parts)[8]


def get_pipe_w_f1_results(pipe, X_train, X_test, y_train, y_test, cfm_path) -> dict:
    f1_scores = cross_val_score(pipe, X_train, y_train, scoring="f1_weighted")
    avg_val_f1 = np.average(f1_scores)
    # print("Average Cross-Validation F1 Score: ", avg_f1)

    pipe.fit(X_train, y_train)
    weighted_prediction = pipe.predict(X_test)
    w_f1 = f1_score(y_test, weighted_prediction, average="weighted")
    # print(f"Weighted F1 Score: {w_f1}")
    plot_confusion_matrix(
        y_test, weighted_prediction, cfm_path, labels=["Large", "Small"]
    )

    return {"val f1": round(avg_val_f1, 3), "test f1": round(w_f1, 3)}


def get_pipe_w_f1_results_shock(
    pipe, X_train, X_test, y_train, y_test, cfm_path
) -> dict:
    f1_scores = cross_val_score(pipe, X_train, y_train, scoring="f1_weighted")
    avg_val_f1 = np.average(f1_scores)
    # print("Average Cross-Validation F1 Score: ", avg_f1)

    pipe.fit(X_train, y_train)
    weighted_prediction = pipe.predict(X_test)
    w_f1 = f1_score(y_test, weighted_prediction, average="weighted")
    # print(f"Weighted F1 Score: {w_f1}")
    plot_confusion_matrix(
        y_test, weighted_prediction, cfm_path, labels=["No Shock", "Shock"]
    )

    return {"val f1": round(avg_val_f1, 3), "test f1": round(w_f1, 3)}


def plot_confusion_matrix(y_true, y_pred, cfm_path, labels):
    cf_matrix = confusion_matrix(y_true, y_pred, labels).astype(int)
    ax = sns.heatmap(cf_matrix, annot=True, cmap="Blues")
    ax.set_title("Actual Outcome vs. Predicted Outcome")
    ax.set_xlabel("\nPredicted Outcomes")
    ax.set_ylabel("Actual Outcomes")

    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    plt.savefig(cfm_path)
    plt.close()


def logistic_regression(
    X_train, X_test, y_train, y_test, results: dict, cfm_path
) -> dict:
    # l2 -> all neurons make small contributions
    # l1 -> only some neurons make large contributions
    pipe = make_pipeline(
        StandardScaler(), LogisticRegression(penalty="l2", max_iter=400),
    )
    clf = LogisticRegression(
        penalty="l2",
        # l1_ratio=0,
        C=50,
        # solver="saga",
        max_iter=500,
        class_weight={"Large": 0.2, "Small": 0.4},
    )

    results["logistic_reg"] = get_pipe_w_f1_results(
        pipe, X_train, X_test, y_train, y_test, cfm_path
    )

    return results


def linear_discriminant(
    X_train, X_test, y_train, y_test, results: dict, cfm_path, dataset_type
) -> dict:

    clf = LinearDiscriminantAnalysis()

    pipe = make_pipeline(StandardScaler(), clf)

    if dataset_type == "Shock Test":
        results["linear_discriminant"] = get_pipe_w_f1_results_shock(
            pipe, X_train, X_test, y_train, y_test, cfm_path
        )
    else:
        results["linear_discriminant"] = get_pipe_w_f1_results(
            pipe, X_train, X_test, y_train, y_test, cfm_path
        )

    return results


def svm_svc(X_train, X_test, y_train, y_test, results: dict, cfm_path) -> dict:
    # l2 -> all neurons make small contributions
    # l1 -> only some neurons make large contributions
    pipe = make_pipeline(StandardScaler(), SVC())

    results["SVC"] = get_pipe_w_f1_results(
        pipe, X_train, X_test, y_train, y_test, cfm_path
    )

    return results


def gaussian_NB(X_train, X_test, y_train, y_test, results: dict, cfm_path) -> dict:
    pipe = make_pipeline(StandardScaler(), PCA(3), GaussianNB())

    results["gaussian_nb"] = get_pipe_w_f1_results(
        pipe, X_train, X_test, y_train, y_test, cfm_path
    )

    return results


# Block -> Outcome Reward ->
# /media/rory/Padlock_DT/BLA_Analysis/Decoding/Arranged_Dataset/{1.0}/[Large]/{BLA-Insc-1}/{RDT D1}/[trail_1.csv]
def binary_classifications():
    ROOT_PATH = Path(r"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Arranged_Dataset/")
    session = "RDT D1"
    mouse = [
        "BLA-Insc-1",
        "BLA-Insc-2",
        "BLA-Insc-3",
        "BLA-Insc-5",
        "BLA-Insc-6",
        "BLA-Insc-7",
        "BLA-Insc-8",
        "BLA-Insc-9",
        "BLA-Insc-11",
        "BLA-Insc-13",
    ]
    blocks = ["1.0", "2.0", "3.0"]

    for i, block in enumerate(blocks):
        if i != 2:
            continue

        print()
        print(f"PREDICTING OUTCOME IN BLOCK {block}, {mouse} {session}")
        files = find_paths(ROOT_PATH, block, mouse, session, "trail")
        new_files = [i for i in files if not "Omission" in i and not "ITI" in i]
        # print(*files, sep="\n")
        print("Number of trials (csvs): ", len(new_files))

        y = []
        X = []
        # to record scores across multiple classifiers
        f1_results = {}

        for csv in new_files:
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

        print("Number of cells: ", len(X))
        # pull only unique elements in list
        labels = []
        for ele in y:
            if ele not in labels:
                labels.append(ele)
        print(f"Labels: {labels}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, train_size=0.80, random_state=None
        )

        ######### INPUT CLASSIFIERS HERE #########
        f1_results = linear_discriminant(X_train, X_test, y_train, y_test, f1_results)
        # f1_results = gaussian_NB(X_train, X_test, y_train, y_test, f1_results)
        # f1_results = svm_svc(X_train, X_test, y_train, y_test, f1_results)

        ######### PRINT RESULTS #########
        for key, val in f1_results.items():
            print(key, ":", val)


# Shock Test
# /media/rory/Padlock_DT/BLA_Analysis/Decoding/Arranged_Dataset/Shock Test/Shock/0.32-0.4/BLA-Insc-1/trial_3.csv
def binary_classifications_shock():
    ROOT_PATH = Path(r"/media/rory/Padlock_DT/BLA_Analysis/Decoding/Arranged_Dataset/")
    mouse = [
        "BLA-Insc-1",
        "BLA-Insc-2",
        "BLA-Insc-3",
        "BLA-Insc-5",
        "BLA-Insc-6",
        "BLA-Insc-7",
        "BLA-Insc-8",
        "BLA-Insc-9",
        "BLA-Insc-11",
        "BLA-Insc-13",
    ]
    shock_intensities = ["0-0.1", "0.12-0.2", "0.22-0.3", "0.32-0.4", "0.42-0.5"]
    norm = True

    for m in mouse:
        for i, shock_intensity in enumerate(shock_intensities):

            print()
            print(f"PREDICTING OUTCOME IN INTENSITIES {shock_intensity}, {m}")
            files = find_paths_shock(ROOT_PATH, shock_intensity, m, "trial")
            # print(*files, sep="\n")
            print("Number of trials (csvs): ", len(files))

            y = []
            X = []
            # to record scores across multiple classifiers
            f1_results = {}

            for csv in files:
                csv = Path(csv)
                outcome = csv.parts[8]
                df: pd.DataFrame
                df = pd.read_csv(csv)

                df = df.T
                df = df.iloc[1:, :]  # remove first row (the cell names)
                # go through columns and add to X and y
                for col in list(df.columns):
                    if norm == True:
                        X.append(stats.zscore(list(df[col])))
                    else:
                        X.append(list(df[col]))
                    y.append(outcome)

            print("Number of cells: ", len(X))
            # pull only unique elements in list
            labels = []
            for ele in y:
                if ele not in labels:
                    labels.append(ele)
            print(f"Labels: {labels}")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, train_size=0.80, random_state=None
            )
            # Confusion matrix save?
            cfm_dir = "/".join(csv.parts[0:8])
            if norm == True:
                cfm_path = os.path.join(cfm_dir, f"norm_cfm_{shock_intensity}_{m}.png")
            else:
                cfm_path = os.path.join(cfm_dir, f"cfm_{shock_intensity}_{m}.png")
            ######### INPUT CLASSIFIERS HERE #########
            f1_results = linear_discriminant(
                X_train, X_test, y_train, y_test, f1_results, cfm_path, "Shock Test",
            )
            # f1_results = gaussian_NB(X_train, X_test, y_train, y_test, f1_results, cfm_path)
            # f1_results = svm_svc(X_train, X_test, y_train, y_test, f1_results, cfm_path)

            ######### PRINT RESULTS #########
            for key, val in f1_results.items():
                print(key, ":", val)


if __name__ == "__main__":
    binary_classifications_shock()

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
