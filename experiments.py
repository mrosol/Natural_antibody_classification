import gc
import logging
import numpy as np
import tensorflow as tf
from NA_train import train_model

logging.basicConfig(level=logging.INFO)
np.random.seed(42)
tf.random.set_seed(42)


def chose_params(parameters_grid: dict, model_type: str) -> dict:
    """Function is randomly choosing parameter values from a given dictionary

    Args:
        parameters_grid (dict): Each key contains a list of possible values for a given parameter (key)
        model_type (str): Type of the model which will be trained (MLP, GRU, LSTM)

    Returns:
        dict: Dictionary with a chosen set of parameter values
    """
    params = {}
    for key in parameters_grid.keys():
        if (model_type == "MLP") & (key in ["lstm_cells", "gru_cells"]):
            pass
        elif (model_type == "GRU") & (key == "lstm_cells"):
            pass
        elif (model_type == "LSTM") & (key == "gru_cells"):
            pass
        else:
            parameters_grid[key]
            params[key] = parameters_grid[key][np.random.randint(0, len(parameters_grid[key]))]
    params["model_type"] = model_type

    return params


if __name__ == "__main__":
    parameters_grid = {
        "embedding_dim": [8, 16, 32],
        "epochs": [10],
        "dense_neurons": [50, 100, 200],
        "batch_size": [256, 512],
        "learning_rate": [0.001, 0.0001],
        "lstm_cells": [10, 20],
        "gru_cells": [10, 20],
    }

    for model_type in ["MLP", "LSTM", "GRU"]:
        used_sets = []
        for idx in range(5):
            was = True
            while was:
                params = chose_params(parameters_grid, model_type)

                if params in used_sets:
                    pass
                else:
                    was = False
                    used_sets.append(params)
            params["run_name"] = f"{model_type}_{idx}"
            logging.info(f"Choosen params set: {params}")
            train_model(**params)
            gc.collect()
