from typing import List, Callable
import pathlib
from sklearn.model_selection import train_test_split
from keras import layers
import pandas as pd
import pickle
import tensorflow as tf
import json

malicious_types = ["phishing", "malware", "malicious"]
clean_types = ["benign"]  # defacement, maybe
allowed_types = malicious_types + clean_types

MALICIOUS_TYPE = "malicious"
CLEAN_TYPE = "benign"
TRAIN_CSV = "train.csv.gz"
TEST_CSV = "test.csv.gz"
VAL_CSV = "val.csv.gz"
DIST_CSV = "distribution.csv"
CAT_FILE = "categories.pkl"
TRAIN_CONFIG_FILE = "model.config.json"


class ModelConfig(object):
    """Represent model configuration dictionary as an object"""

    def __init__(self, name: str, data: dict) -> None:
        self.batch_size = data.get("batch_size", 64)
        self.max_features = data.get("max_features", 128)
        self.split = data.get("split", "character")
        self.standardize = data.get(
            "standardize", "lower_and_strip_punctuation")
        self.max_length = data.get("max_length", 2048)
        self.model = data.get("model")
        self.stringify = data.get("stringify", True)
        self.name = name
        self.output_path = data.get("output_path")
        self.epoch = data.get("epoch", 30)
        self.patience = data.get("patience", 10)

    @staticmethod
    def load(filename: str) -> dict:
        """Load configurations from .json file."""
        with open(filename, "r") as f:
            # Load the JSON data from file
            data = json.load(f)

        configs = []
        # For each model configuration, append it to the list.
        for model_name, config in data.items():
            configs.append(ModelConfig(model_name, config))
        return configs


def prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Clean chunk from duplicate data and incorrect types."""
    chunk.dropna(inplace=True)
    chunk.drop_duplicates(inplace=True)
    chunk = chunk.loc[chunk["type"].isin(allowed_types), ["url", "type"]]
    chunk.loc[chunk["type"].isin(clean_types), "type"] = CLEAN_TYPE
    chunk.loc[chunk["type"].isin(malicious_types), "type"] = MALICIOUS_TYPE
    return chunk


def fuse_datasets(output_file: str, filenames: List[str], chunksize: int) -> None:
    """Merge multiple datasets into single large dataset."""
    pathlib.Path(output_file).unlink(missing_ok=True)

    for i, filename in enumerate(filenames):
        chunks = pd.read_csv(filename, chunksize=chunksize)
        chunk: pd.DataFrame
        for j, chunk in enumerate(chunks):
            chunk = prepare_chunk(chunk)
            # The first chunk has to save header, so it can be inferred by pd.read_csv
            if i == 0 and j == 0:
                chunk.to_csv(output_file, mode="a", header=True, index=False)
            else:
                chunk.to_csv(output_file, mode="a", header=False, index=False)


def split_dataset(output_folder: str, filename: str, train_ratio: float, validation_ratio: float, test_ratio: float):
    """Split dataset into training, validation and testing data. All parts are saved in own file."""
    # Make sure all ratios sums up to 100%.
    assert(train_ratio + validation_ratio + test_ratio == 1)
    out = pathlib.Path(output_folder)
    out.mkdir(parents=True, exist_ok=False)

    # Get dataset frame
    df = pd.read_csv(filename)

    # Count data by their type
    counts = df.groupby(by="type").count()
    malicious = counts.loc[MALICIOUS_TYPE, "url"]
    clean = counts.loc[CLEAN_TYPE, "url"]

    # Downsample minority class
    df_clean: pd.DataFrame
    df_malicious: pd.DataFrame
    if malicious < clean:
        df_clean = df.loc[df["type"] == CLEAN_TYPE, :].sample(n=malicious)
        df_malicious = df.loc[df["type"] == MALICIOUS_TYPE, :]
    else:
        df_clean = df.loc[df["type"] == MALICIOUS_TYPE, :].sample(n=malicious)
        df_malicious = df.loc[df["type"] == CLEAN_TYPE, :]

    # Create new balanced dataset
    df = pd.concat([df_clean, df_malicious], ignore_index=True)

    # Extract categories from type column
    categories = df["type"].unique()
    type_labels = pd.Categorical(
        df["type"], categories=categories, ordered=False).codes
    df["type"] = type_labels

    # Save categories to files, so reverse mapping can be done later
    categories: List[str] = categories.tolist()
    with open(out/CAT_FILE, "wb") as f:
        pickle.dump(categories, f)

    clean_index = categories.index(CLEAN_TYPE)
    malicious_index = categories.index(MALICIOUS_TYPE)

    # Work with test and validation part as one whole part for now
    test_val_ratio = test_ratio + validation_ratio

    x_train, x_test, y_train, y_test = train_test_split(
        df["url"], df["type"], test_size=test_val_ratio, random_state=1337, shuffle=True, stratify=df["type"])

    # Now, break the  two parts into ratios
    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=validation_ratio / test_val_ratio, random_state=1337, shuffle=True, stratify=y_test)

    # Create 3 mentioned datasets
    data_stats = []
    types = [TRAIN_CSV, TEST_CSV, VAL_CSV]
    for label, (x, y) in zip(types, [(x_train, y_train), (x_test, y_test), (x_val, y_val)]):
        # Create dataset from url and type
        data = pd.concat([x, y], axis=1)

        # Create count statistics again. Thats for evaluation purposes.
        stats = data.groupby(by="type").count()
        data.to_csv(out/label, header=True)
        data_stats.append([stats.loc[clean_index, "url"],
                          stats.loc[malicious_index, "url"]])
    data_stats = pd.DataFrame(data_stats, columns=categories, index=types)
    print(data_stats)
    data_stats.to_csv(out/DIST_CSV)


def load_train_data(data_folder: str) -> pd.DataFrame:
    """Load train data from the dataset folder."""
    data_folder = pathlib.Path(data_folder)
    return pd.read_csv(pathlib.Path(data_folder)/TRAIN_CSV)


def load_test_data(data_folder: str) -> pd.DataFrame:
    """Load test data from the dataset folder."""
    data_folder = pathlib.Path(data_folder)
    return pd.read_csv(pathlib.Path(data_folder)/TEST_CSV)


def load_validation_data(data_folder: str) -> pd.DataFrame:
    """Load validation data from the dataset folder."""
    data_folder = pathlib.Path(data_folder)
    return pd.read_csv(pathlib.Path(data_folder)/VAL_CSV)


def load_dist_data(data_folder: str) -> pd.DataFrame:
    """Load distribution data from the dataset folder."""
    data_folder = pathlib.Path(data_folder)
    return pd.read_csv(pathlib.Path(data_folder)/DIST_CSV)


def load_category_translator(data_folder: str) -> Callable[[int], str]:
    """Load reverse category mapping (int -> str) from the dataset folder."""
    data_folder = pathlib.Path(data_folder)
    categories: List[str]
    with open(data_folder/CAT_FILE, "rb") as f:
        categories = pickle.load(f)

    return lambda index: categories[index]


def load_as_tensor(file: str, max_length: int, batch_size: int):
    """Load data from file as Tensorflow tensor"""
    df = pd.read_csv(file)
    df["url"] = df["url"].apply(lambda url: url[:max_length])
    return tf.data.Dataset.from_tensor_slices(
        (df["url"].values, df["type"].values)).batch(batch_size)


def load_vectorized_tensors(data_folder: str, config: ModelConfig):
    """
    Load vectorized tensors from dataset folder and return all data
    plus vectorization layer for model stringify.
    """
    data_folder = pathlib.Path(data_folder)
    raw_train_ds = load_as_tensor(
        data_folder/TRAIN_CSV, config.max_length, config.batch_size)
    raw_test_ds = load_as_tensor(
        data_folder/TEST_CSV, config.max_length, config.batch_size)
    raw_val_ds = load_as_tensor(
        data_folder/VAL_CSV, config.max_length, config.batch_size)

    vectorization_layer = layers.TextVectorization(
        max_tokens=config.max_features,
        vocabulary=None,
        standardize=config.standardize,
        output_mode="int",
        output_sequence_length=config.max_length,
        split=config.split
    )

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorization_layer(text), label

    print("training vectorization embedding")
    # Train word vectorizer
    text_ds = raw_train_ds.map(lambda x, _: x)
    vectorization_layer.adapt(text_ds)

    print("vectorizing urls")
    # Prepare word vectors
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    print("vectorized all urls\ncreating cache")
    # Make use of cache, so the training process is more efficient
    train_ds = train_ds.cache().prefetch(buffer_size=10)
    val_ds = val_ds.cache().prefetch(buffer_size=10)
    test_ds = test_ds.cache().prefetch(buffer_size=10)

    return (train_ds, val_ds, test_ds, vectorization_layer)
