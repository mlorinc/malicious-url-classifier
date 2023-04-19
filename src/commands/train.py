import importlib
import pathlib
import tensorflow as tf
from keras import layers
from .dataset import load_vectorized_tensors, ModelConfig
import keras
import time
import pandas as pd
import json

CHECKPOINT_FOLDER = "checkpoint.model"
BEST_MODEL_FOLDER = "best.model"
HISTORY_CSV = "history.csv.gz"
CONFIG_JSON = "config.json"


class TimeHistory(keras.callbacks.Callback):
    """Measure duration of each epoch."""

    def on_train_begin(self, logs={}):
        """Initialize time array."""
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        """Remember epoch start time."""
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        """Take measurement of epoch duration"""
        self.times.append(time.time() - self.epoch_time_start)


def scheduler(epoch, lr):
    if epoch < 10 or epoch % 10 != 0:
        return lr
    else:
        return lr / 10


def import_model(config: ModelConfig) -> keras.Model:
    """Import experimental model from models module."""
    model_package = importlib.import_module(config.model, "models")
    return model_package.get_model(config)


def train(dataset_path: str, config: ModelConfig):
    """
    Train NN on selected dataset. Training accuracy, loss, validation accuracy,
    validation accuracy, validation loss, epoch durations will be stored with the best
    model. 
    """

    # Create output folder
    model_root = pathlib.Path(config.output_path)
    model_root.mkdir(parents=True, exist_ok=False)

    # Import experimental model and load dataset data
    model = import_model(config)
    train_ds, val_ds, test_ds, vectorization_layer = load_vectorized_tensors(
        dataset_path, config)

    # Make sure, the best model is saved.
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_root/CHECKPOINT_FOLDER,
        save_weights_only=False,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True)

    # Do not train NN, which do not perform well.
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config.patience)
    time_callback = TimeHistory()

    history = model.fit(train_ds, epochs=config.epoch, validation_data=val_ds, callbacks=[
                        time_callback,
                        keras.callbacks.LearningRateScheduler(scheduler),
                        model_checkpoint_callback,
                        early_stopping_callback])

    # Save history with epoch times.
    stats = pd.DataFrame(data=history.history)
    stats["time"] = time_callback.times
    stats.to_csv(model_root/HISTORY_CSV)
    with open(model_root/CONFIG_JSON, "w") as f:
        json.dump(vars(config), f)

    # Make model accept strings instead word vectors
    if config.stringify:
        string_model = create_string_model(model, vectorization_layer)
        string_model.save(model_root/BEST_MODEL_FOLDER)
    else:
        model.save(model_root/BEST_MODEL_FOLDER)


def create_string_model(model: keras.Model, vectorization_layer: layers.TextVectorization) -> keras.Model:
    # A string input
    inputs = keras.Input(shape=(1,), dtype="string")
    # Turn strings into vocab indices
    indices = vectorization_layer(inputs)
    # Turn vocab indices into predictions
    outputs = model(indices)

    # Our end to end model
    end_to_end_model = keras.Model(inputs, outputs)
    end_to_end_model.compile(
        loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy"]
    )

    return end_to_end_model


def get_history(model_root: pathlib.Path) -> pd.DataFrame:
    """Get NN training history as dataframe."""
    return pd.read_csv(model_root/HISTORY_CSV)


def get_model(model_root: pathlib.Path) -> pd.DataFrame:
    """Load model from the model folder."""
    return keras.models.load_model(model_root/BEST_MODEL_FOLDER)


def get_checkpoint_model(model_root: pathlib.Path) -> pd.DataFrame:
    """Get checkpoint model from the model folder."""
    return keras.models.load_model(model_root/CHECKPOINT_FOLDER)


def get_config(model_root: pathlib.Path) -> ModelConfig:
    """Get config used during training."""
    with open(model_root/CONFIG_JSON, "r") as f:
        # Load the JSON data from file
        config = json.load(f)
    return ModelConfig(config["name"], config)
