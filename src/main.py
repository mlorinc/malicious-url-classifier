import argparse

import sys

import keras
import pandas as pd
import sklearn as sk
import scipy as sp
import tensorflow as tf
import platform

from commands.dataset import fuse_datasets, split_dataset, ModelConfig
from commands.evaluate import evaluate
from commands.train import train, create_string_model

def parser_arguments():
    # Create the parser and add subparsers for the three commands
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    fuse_parser = subparsers.add_parser("dataset:fuse", help="Fuse multiple datasets into one")
    fuse_parser.add_argument("output_file", help="File where fused dataset will be saved")
    fuse_parser.add_argument("datasets", nargs="+", type=str, help="List of input datasets in csv format")
    fuse_parser.add_argument("--chunksize", type=int, default=4096*4, help="Read chunk size")

    split_parser = subparsers.add_parser("dataset:split", help="Split dataset into train, validation and testing part")
    split_parser.add_argument("output_folder", help="Output folder where datasets will be stored")
    split_parser.add_argument("input_file", help="Input dataset in csv format")
    split_parser.add_argument("train_ratio", type=float, help="Training data ratio")
    split_parser.add_argument("validation_ratio", type=float, help="Validation data ratio")
    split_parser.add_argument("test_ratio", type=float, help="Testing data ratio")

    train_parser = subparsers.add_parser("model:train", help="Train a machine learning model")
    train_parser.add_argument("dataset", help="Path to the dataset for training/validation")
    train_parser.add_argument("config", help="Model configuration")

    predict_parser = subparsers.add_parser("model:predict", help="Make a prediction using a trained model")
    predict_parser.add_argument("model_path", help="Name of the machine learning model")

    predict_parser = subparsers.add_parser("model:evaluate", help="Evaluate model and create statistics")
    predict_parser.add_argument("output_folder", help="Output directory where figures will be stored")
    predict_parser.add_argument("dataset", help="Path to the dataset for testing")
    predict_parser.add_argument("model_path", nargs="+", type=str, help="Path of the machine learning models")
    predict_parser.add_argument("--format", type=str, action="append", default=["pdf"], help="Output figure format. Default is a pdf.")

    # Parse the arguments and call the appropriate function
    return parser.parse_args()

def main():
    args = parser_arguments()

    print(f"Python Platform: {platform.platform()}")
    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Keras Version: {keras.__version__}")
    print()
    print(f"Python {sys.version}")
    print(f"Pandas {pd.__version__}")
    print(f"Scikit-Learn {sk.__version__}")
    print(f"SciPy {sp.__version__}")
    gpu = len(tf.config.list_physical_devices("GPU"))>0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")

    if args.command == "dataset:fuse":
        fuse_datasets(args.output_file, args.datasets, args.chunksize)
    elif args.command == "dataset:split":
        split_dataset(args.output_folder, args.input_file, args.train_ratio, args.validation_ratio, args.test_ratio)
    elif args.command == "model:train":
        dataset = args.dataset
        configs = ModelConfig.load(args.config)
        for config in configs:
            print(f"Training: {config.name}")
            train(dataset, config)
    elif args.command == "model:evaluate":
        evaluate(args.model_path, args.output_folder, args.dataset, args.format)
main()