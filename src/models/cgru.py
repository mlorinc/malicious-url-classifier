import keras
from keras import layers
from commands.dataset import ModelConfig

def get_model(config: ModelConfig):
    inputs = keras.Input(shape=(None,), dtype="int64", name="text")
    x = layers.Embedding(config.max_features, 128)(inputs)
    split = layers.Dropout(0.5)(x)
    c1 = layers.Conv1D(128, 2, padding="valid", activation="relu", strides=1)(split)
    c2 = layers.Conv1D(128, 4, padding="valid", activation="relu", strides=1)(split)
    merged = layers.Concatenate(axis=1)([c1, c2])

    x = layers.GRU(64, return_sequences=False)(merged)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    model = keras.Model(inputs, predictions)

    # Compile the model with binary crossentropy loss and an adam optimizer.
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy"])
    return model