import pandas as pd
import seaborn as sns
import keras
import matplotlib.pyplot as plt
import pathlib
import time
from typing import List
from .train import get_config, get_history, get_model
from .dataset import load_test_data, load_category_translator, MALICIOUS_TYPE, CLEAN_TYPE
from sklearn.metrics import DetCurveDisplay

CLEAN_CLASS = 1
MALICIOUS_CLASS = 0


def evaluate(model_paths: List[str], output_folder: str, dataset: str, formats: str):
    """
    Evaluate models on test dataset and save plots to output folder.
    Multiple save format can be used e.g. pdf, png or jpg.
    """

    # Prepare file system structure
    model_paths: List[pathlib.Path] = [pathlib.Path(p) for p in model_paths]
    output_folder: pathlib.Path = pathlib.Path(output_folder)
    dataset: pathlib.Path = pathlib.Path(dataset)

    if not dataset.exists():
        raise ValueError(f"dataset \"{dataset}\" does not exist")

    output_folder.mkdir(parents=True, exist_ok=False)

    # Initialize dataframes which hold overall statistics of all models
    df_history_stats = pd.DataFrame()
    df_stats = pd.DataFrame()

    # Evaluate each model
    for model_path in model_paths:
        # Create folder for each model, so plots are nicely grouped by models
        model_output_folder = output_folder/model_path
        model_output_folder.mkdir(parents=True, exist_ok=False)

        # Load required assets
        df_history = get_history(model_path)
        df_test = load_test_data(dataset)
        config = get_config(model_path)
        model: keras.Model = get_model(model_path)
        to_label = load_category_translator(dataset)

        # Measure inference time and get validation data
        start = time.time()
        df_test["predicted"] = model.predict(
            df_test["url"].values, config.batch_size)
        end = time.time()

        # Convert predicted soft values to hard values
        df_test["predicted_label"] = df_test["predicted"].apply(
            lambda p: MALICIOUS_TYPE if p >= 0.5 else CLEAN_TYPE)
        # Map numeric values to labels, so they are nicely represented in confusion matrix
        df_test["actual"] = df_test["type"].apply(to_label)

        # Create confusion matrix
        df_confusion = pd.crosstab(df_test["actual"], df_test["predicted_label"], rownames=[
                                   "Actual"], colnames=["Predicted"])

        # Append the model history into overall history
        df_history = df_history.reset_index(names=["epoch"])
        df_history.dropna(inplace=True)
        df_history.loc[:, "model_name"] = config.name
        df_history_stats = pd.concat([df_history_stats, df_history])

        # Save time and test accuracy into overall stats
        df_stats.loc[config.name, "time"] = end - start
        df_stats.loc[config.name, "test_accuracy"] = len(
            df_test[df_test["predicted_label"] == df_test["actual"]].index) / len(df_test.index)

        # Create plots for each format
        for format in formats:
            plt.figure(figsize=(10, 7))
            sns.heatmap(data=df_confusion, annot=True, cmap="Blues", fmt="g")
            plt.savefig(model_output_folder/f"confusion.{format}")
            plt.clf()

            ax = sns.lineplot(data=df_history, x="epoch", y="accuracy")
            ax.set(xlabel="Epoch", ylabel="Accuracy")
            plt.savefig(model_output_folder/f"accuracy.{format}")
            plt.clf()

            ax = sns.lineplot(data=df_history, x="epoch", y="loss")
            ax.set(xlabel="Epoch", ylabel="Loss")
            plt.savefig(model_output_folder/f"loss.{format}")
            plt.clf()

            ax = sns.lineplot(data=df_history, x="epoch", y="val_loss")
            ax.set(xlabel="Epoch", ylabel="Validation loss")
            plt.savefig(model_output_folder/f"val_loss.{format}")
            plt.clf()

            ax = sns.lineplot(data=df_history, x="epoch", y="val_accuracy")
            ax.set(xlabel="Epoch", ylabel="Validation accuracy")
            plt.savefig(model_output_folder/f"val_accuracy.{format}")
            plt.clf()

            ax = sns.lineplot(data=df_history, x="epoch", y="time")
            ax.set(xlabel="Epoch", ylabel="Time in seconds")
            plt.savefig(model_output_folder/f"time.{format}")
            plt.clf()

            fig, ax_det = plt.subplots(1, 1, figsize=(11, 5))
            DetCurveDisplay.from_predictions(
                df_test["actual"], df_test["predicted"], pos_label=MALICIOUS_TYPE, ax=ax_det, name=config.name)
            fig.savefig(model_output_folder/f"det.{format}")
            plt.close(fig)

    # Create overall plots for each format
    for format in formats:
        ax = sns.lineplot(data=df_history_stats, x="epoch",
                          y="accuracy", hue="model_name")
        ax.set(xlabel="Epoch", ylabel="Accuracy")
        plt.savefig(output_folder/f"accuracy.{format}")
        plt.clf()

        ax = sns.lineplot(data=df_history_stats, x="epoch",
                          y="loss", hue="model_name")
        ax.set(xlabel="Epoch", ylabel="Loss")
        plt.savefig(output_folder/f"loss.{format}")
        plt.clf()

        ax = sns.lineplot(data=df_history_stats, x="epoch",
                          y="val_loss", hue="model_name")
        ax.set(xlabel="Epoch", ylabel="Validation loss")
        plt.savefig(output_folder/f"val_loss.{format}")
        plt.clf()

        ax = sns.lineplot(data=df_history_stats, x="epoch",
                          y="val_accuracy", hue="model_name")
        ax.set(xlabel="Epoch", ylabel="Validation accuracy")
        plt.savefig(output_folder/f"val_accuracy.{format}")
        plt.clf()

        # Measure training time of each model
        training_time_df = df_history_stats.groupby(
            by="model_name").sum().sort_values(by="time", ascending=True)
        # Remove the model from dataset as it was incorrectly measured; todo: remove
        training_time_df.drop(index="cnn-lstm-128", inplace=True)

        ax = sns.barplot(data=training_time_df,
                         y=training_time_df.index, x="time")
        ax.set(ylabel="Model", xlabel="Time in seconds")
        ax.bar_label(ax.containers[0])
        sns.despine(left=True, bottom=True)
        plt.savefig(output_folder/f"time.{format}")
        plt.clf()

        df_time_stats = df_stats.sort_values(by="time", ascending=True)
        ax = sns.barplot(data=df_time_stats, y=df_time_stats.index, x="time")
        ax.set(ylabel="Model", xlabel="Time in seconds")
        ax.bar_label(ax.containers[0])
        sns.despine(left=True, bottom=True)
        plt.savefig(output_folder/f"inference-time.{format}")
        plt.clf()

        df_test_accuracy_stats = df_stats.sort_values(
            by="test_accuracy", ascending=False)
        print(df_test_accuracy_stats)
        ax = sns.barplot(data=df_test_accuracy_stats,
                         y=df_test_accuracy_stats.index, x="test_accuracy")
        ax.set(ylabel="Model", xlabel="Accuracy")
        ax.bar_label(ax.containers[0])
        sns.despine(left=True, bottom=True)
        plt.savefig(output_folder/f"test-accuracy.{format}")
        plt.clf()
