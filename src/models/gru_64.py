import keras
from keras import layers
from commands.dataset import ModelConfig

def get_model(config: ModelConfig):
    inputs = keras.Input(shape=(None,), dtype="int64", name="text")
    x = layers.Embedding(config.max_features, 64)(inputs)
    x = layers.GRU(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    model = keras.Model(inputs, predictions)

    # Compile the model with binary crossentropy loss and an adam optimizer.
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy"])
    return model