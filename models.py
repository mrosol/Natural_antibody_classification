import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, Bidirectional, Flatten
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import History
from utils import calculate_metrics


class Model:
    def __init__(self, embedding_dim, words_number, max_length):
        model = Sequential()
        model.add(Embedding(words_number + 1, embedding_dim, input_length=max_length))

        self.model = model

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        run_name: str,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> History:
        """ Function for fitting the model with the given parameters. The model itself is a parameter of the Model class.

        Args:
            X_train (pd.DataFrame): Tokenized features
            y_train (pd.Series): Labels
            epochs (int): Number of epochs for fitting
            learning_rate (float): Learning rate
            batch_size (int): Batch size
            run_name (str): Name of the run for the MLflow and Tensorboard logging
            X_val (pd.DataFrame, optional): Tokenized validation data. Defaults to None.
            y_val (pd.Series, optional): Labels for the validation data. Defaults to None.

        Returns:
            History: History of fitting.
        """
        self.batch_size = batch_size
        if run_name:
            log_dir = "logs/fit/" + run_name
        else:
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # Early stopping callback
        stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.001, patience=3
        )
        opt = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=opt,
            loss="binary_crossentropy",
            metrics=[
                metrics.BinaryAccuracy(name="accuracy"),
                metrics.AUC(name="AUC"),
                metrics.Precision(name="precision"),
                metrics.Recall(name="recall"),
            ],
        )
        # Validation data is used only when passed as an argument
        val_data = (X_val, y_val) if X_val is not None else None
        hist = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=[tensorboard_callback, stopping_callback],
        )

        return hist

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float, float, float, float]:
        """ Function for evaluation of the model returning differnet metrics

        Args:
            X (pd.DataFrame): tokenized features data
            y (pd.Series): labels

        Returns:
            Tuple[float, float, float, float, float]: accuracy, precision, recall, specificity, f1 score, AUC
        """

        y_hat = self.model.predict(X, batch_size=self.batch_size)

        accuracy, precision, recall, specificity, f1, auc_metric = calculate_metrics(y, y_hat)

        return accuracy, precision, recall, specificity, f1, auc_metric

    def predict(self, X: pd.DataFrame) -> np.array:
        """ Predicting the score (human vs. mouse) for the given data

        Args:
            X (pd.DataFrame): Tokenized data

        Returns:
            np.array: Scores - closer to 0 - mouse, closer to 1 - human
        """
        y_hat = self.model.predict(X, batch_size=self.batch_size)

        return y_hat


class Model_LSTM(Model):
    def __init__(self, lstm_cells, dense_neurons, embedding_dim, words_number, max_length):
        super().__init__(embedding_dim, words_number, max_length)

        self.model.add(
            Bidirectional(LSTM(lstm_cells), merge_mode="concat", weights=None, backward_layer=None)
        )
        self.model.add(Dense(dense_neurons, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))


class Model_GRU(Model):
    def __init__(self, gru_cells, dense_neurons, embedding_dim, words_number, max_length):
        super().__init__(embedding_dim, words_number, max_length)

        self.model.add(
            Bidirectional(GRU(gru_cells), merge_mode="concat", weights=None, backward_layer=None)
        )
        self.model.add(Dense(dense_neurons, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))


class Model_MLP(Model):
    def __init__(self, dense_neurons, embedding_dim, words_number, max_length):
        super().__init__(embedding_dim, words_number, max_length)
        self.model.add(Flatten())
        self.model.add(Dense(dense_neurons, activation="relu"))
        self.model.add(Dense(dense_neurons, activation="relu"))
        self.model.add(Dense(dense_neurons, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))
