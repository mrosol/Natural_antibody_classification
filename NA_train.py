import logging
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras
import click

from utils import *
from models import *

np.random.seed(42)
tf.random.set_seed(42)
logging.basicConfig(level=logging.INFO)


def log_params(
    embedding_dim: int,
    learning_rate: float,
    dense_neurons: int,
    epochs: int,
    batch_size: int,
    model_type: str,
    lstm_cells: int,
    gru_cells: int,
):
    """Function for logging the training parameters using MLflow.

    Args:
        embedding_dim (int): Dimension of the dense embedding.
        learning_rate (float): Learning rate
        dense_neurons (int): Number of neurons in fully connected layers.
        epochs (int): Number of epochs
        batch_size (int): Batch size
        model_type (str): Model architecture (LSTM/GRU/MLP)
        lstm_cells (int): Number of LSTM cells in bidirectional layer (only if model_type is 'LSTM').
        gru_cells (int): Number of GRU cells in bidirectional layer (only if model_type is 'GRU').
    """
    mlflow.log_param("Embedding dimension", embedding_dim)
    mlflow.log_param("learning_rate", learning_rate)
    if model_type == "LSTM":
        mlflow.log_param("LSTM cells", lstm_cells)
    elif model_type == "GRU":
        mlflow.log_param("GRU cells", gru_cells)
    mlflow.log_param("Dense neurons", dense_neurons)
    mlflow.log_param("Epochs", epochs)
    mlflow.log_param("Batch size", batch_size)


def log_metrics(
    accuracy: float,
    precision: float,
    recall: float,
    specificity: float,
    f1: float,
    auc_metric: float,
    sufix: str,
):
    """Function for logging the metrics of a trained model using MLflow.

    Args:
        accuracy (float): Accuracy (TP+TN)/(TP+TN+FN+FP)
        precision (float): Precision TP/(TP+FP)
        recall (float): Recall TP/(TP+FN)
        specificity (float): Specificty TN/(TN+FP)
        f1 (float): F1 score 2TP/(2TP+FP+FN)
        auc_metric (float): AUC from ROC curve
        sufix (str): Sufix for the stored metrics (train/val/test)
    """

    mlflow.log_metric(f"accuracy_{sufix}", accuracy)
    mlflow.log_metric(f"precision_{sufix}", precision)
    mlflow.log_metric(f"recall_{sufix}", recall)
    mlflow.log_metric(f"specificity_{sufix}", specificity)
    mlflow.log_metric(f"AUC_{sufix}", auc_metric)
    mlflow.log_metric(f"f1_{sufix}", f1)


# @click.command()
# @click.option("--model-type", help="Model type", type=click.Choice(['LSTM', 'GRU', 'MLP']), default=None, required = True)
# @click.option("--embedding-dim", help="Embedding dimension", default=16, type=int)
# @click.option("--dense-neurons", help="Number of neurons in dense layer", default=100, type=int)
# @click.option("--epochs", help="Number of training epochs", default=10, type=int)
# @click.option("--batch-size", help="Number of samples in batch", default=256, type=int)
# @click.option("--learning-rate", help="Learning rate", default=0.001, type=float)
# @click.option("--lstm-cells", help="Number of LSTM cells in bidirectional layer", default=10, type=int)
# @click.option("--gru-cells", help="Number of GRU cells in bidirectional layer", default=10, type=int)
# @click.option("--test", help="Defines if evaluation metrics should be calculated also on a test set after training. Default is False and metrics are calculated on training and validation sets.", default=False, type=bool)


def train_model(
    model_type: str,
    embedding_dim: int = 16,
    dense_neurons: int = 100,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    lstm_cells: int = 10,
    gru_cells: int = 10,
    test: bool = False,
    run_name: str = None,
) -> Model:
    """This function is traing the model with the given model type (architecutre) and parameters.

    Args:
        model_type (str): Defines model architecture (LSTM/GRU/MLP)
        embedding_dim (int, optional): Dimension of the dense embedding. Defaults to 16.
        dense_neurons (int, optional): Number of neurons in fully connected layers. Defaults to 100.
        epochs (int, optional): Number of epochs. Defaults to 10.
        batch_size (int, optional): Batch size. Defaults to 256.
        learning_rate (float, optional): Learning rate. Defaults to 0.001.
        lstm_cells (int, optional): Number of LSTM cells in bidirectional layer (only if model_type is 'LSTM'). Defaults to 10.
        gru_cells (int, optional): Number of GRU cells in bidirectional layer (only if model_type is 'GRU'). Defaults to 10.
        test (bool, optional): If True the model is trained on combined training and validation sets and tested on a test set,
            otherwise the model is trained only on a training set and evaluated on a validation set. Defaults to False.
        run_name (str, optional): Name of the run for the MLflow and Tensorboard logging. Defaults to None.

    Returns:
        Model: Fitted model, which is also stored using MLflow along with all metrics.
    """
    df = load_data()
    logging.info("Data has been loaded")
    tokenizer, df_t = tokenize(df)
    logging.info("Data has been tokenized")
    words_number = len(tokenizer.word_index)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_t)
    logging.info("Data has been split into train, validation and test sets")
    max_length = 150

    if model_type == "LSTM":
        model = Model_LSTM(lstm_cells, dense_neurons, embedding_dim, words_number, max_length)
    elif model_type == "GRU":
        model = Model_GRU(gru_cells, dense_neurons, embedding_dim, words_number, max_length)
    else:  # MLP
        model = Model_MLP(dense_neurons, embedding_dim, words_number, max_length)
    logging.info("Model has been created")

    with mlflow.start_run(run_name=run_name):
        if test:
            X_train_val = X_train.append(X_val)
            y_train_val = y_train.append(y_val)
            hist = model.fit(X_train_val, y_train_val, epochs, learning_rate, batch_size, run_name)
            logging.info("Model has been fitted")
            accuracy, precision, recall, specificity, f1, auc_metric = model.evaluate(
                X_train_val, y_train_val
            )
            log_metrics(accuracy, precision, recall, specificity, f1, auc_metric, "train")
            accuracy, precision, recall, specificity, f1, auc_metric = model.evaluate(
                X_test, y_test
            )
            log_metrics(accuracy, precision, recall, specificity, f1, auc_metric, "test")
            logging.info("Merics have been logged")
        else:
            hist = model.fit(
                X_train, y_train, epochs, learning_rate, batch_size, run_name, X_val, y_val
            )
            logging.info("Model has been fitted")
            accuracy, precision, recall, specificity, f1, auc_metric = model.evaluate(
                X_train, y_train
            )
            log_metrics(accuracy, precision, recall, specificity, f1, auc_metric, "train")
            accuracy, precision, recall, specificity, f1, auc_metric = model.evaluate(X_val, y_val)
            log_metrics(accuracy, precision, recall, specificity, f1, auc_metric, "val")
            logging.info("Merics have been logged")

        epochs_final = len(hist.history["loss"])
        log_params(
            embedding_dim,
            learning_rate,
            dense_neurons,
            epochs_final,
            batch_size,
            model_type,
            lstm_cells,
            gru_cells,
        )
        logging.info("Parameters has been logged")

        mlflow.keras.log_model(model.model, f"model_{run_name}")
        logging.info("Model has been logged")

    return model


if __name__ == "__main__":
    train_model()
