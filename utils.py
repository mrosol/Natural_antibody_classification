from typing import Tuple
import tensorflow as tf
import pandas as pd
import numpy as np
import io
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
)


def load_data() -> pd.DataFrame:
    """Function for loading the data for both human and mouse

    Returns:
        pd.DataFrame: dataframe with two columns, first one contains the sequence and second one the labels (0 - nonhuman, 1 - human)
    """

    df_h1 = pd.read_csv("./data/sample/human_train.txt", header=None)
    df_h2 = pd.read_csv("./data/sample/human_val.txt", header=None)
    df_h3 = pd.read_csv("./data/sample/human_test.txt", header=None)
    df_m = pd.read_csv("./data/sample/mouse_test.txt", header=None)

    # Concatenating all human data
    df_h = pd.concat([df_h1, df_h2, df_h3])

    # Adding labels
    df_h["label"] = 1
    df_m["label"] = 0

    # Concatenating human and mouse data
    df = pd.concat([df_h, df_m])

    df = df.rename(columns={0: "sequence"})

    return df


def tokenize(df: pd.DataFrame) -> Tuple[Tokenizer, pd.DataFrame]:
    """Function is changing a given sequence of chars to integers

    Args:
        df (pd.DataFrame): Loaded data with amino acid sequences

    Returns:
        Tuple[Tokenizer, pd.DataFrame]: Tokenizer contains the information how was the data transformed from chars to int
            Dataframe contains the transformed data.
    """

    try:
        # If there is alread saved tokenizer
        tokenizer = load_tokenizer()
    except:
        tokenizer = Tokenizer(char_level=True)
        tokenizer.fit_on_texts(df["sequence"].values)

        # Saving the created tokenizer for the reproducibility of the results
        # and to enable predictions on a new data
        tokenizer_json = tokenizer.to_json()
        with io.open("tokenizer.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    # Represent input data as word rank number sequences
    X = tokenizer.texts_to_sequences(df["sequence"].values)
    X = pad_sequences(X, maxlen=150)

    df_t = pd.DataFrame(X)
    df_t["label"] = df["label"].values

    return tokenizer, df_t


def train_val_test_split(df: pd.DataFrame) -> Tuple:
    """Function is splitting the data into train (60%), validation (20%) and test (20%) sets
    Args:
        df (pd.DataFrame): Tokenized data with labels

    Returns:
        Tuple: _description_
    """

    train, validate, test = np.split(
        df.sample(frac=1, random_state=42), [int(0.6 * len(df)), int(0.8 * len(df))]
    )

    # Separating features and labels
    X_train, y_train = get_X_y(train)
    X_val, y_val = get_X_y(validate)
    X_test, y_test = get_X_y(test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Function is separting features and labels from the prepared dataframe with tokenized data

    Args:
        df (pd.DataFrame): dataframe with tokenized data and labels

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Features and labels
    """
    X = df.iloc[:, :-1]
    y = df["label"]

    return X, y


def calculate_metrics(y_true: np.array, y_hat: np.array) -> Tuple:
    """Cacluating matrics based on a true labels and output of the model

    Args:
        y_true (np.array): labels (0 or 1)
        y_hat (np.array): Output of the model (values range from 0 to 1)

    Returns:
        Tuple: calculated metrics - accuracy, precision, recall, specificity, f1 score, AUC
    """
    y_pred = np.round(y_hat)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    fpr, tpr, thresholds = roc_curve(y_true, y_hat, pos_label=1)
    auc_metric = auc(fpr, tpr)

    return accuracy, precision, recall, specificity, f1, auc_metric


def load_tokenizer() -> Tokenizer:
    """Loading the existing tokenizer

    Returns:
        Tokenizer: Previously used tokenizer
    """

    with open("tokenizer.json") as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)

    return tokenizer
