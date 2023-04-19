import keras
from keras import layers
from commands.dataset import ModelConfig

def get_model(config: ModelConfig):
    inputs = keras.Input(shape=(None,), dtype="int64")
    x = layers.Embedding(config.max_features, 128)(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv1D(128, 2, padding="valid", activation="relu", strides=1)(x)
    x = layers.Conv1D(128, 4, padding="valid", activation="relu", strides=1)(x)
    x = layers.GRU(64, return_sequences=True)(x)
    x = layers.GRU(64, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    model = keras.Model(inputs, predictions)

    # Compile the model with binary crossentropy loss and an adam optimizer.
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy"])
    return model