<br />
<div align="center">
  <h3 align="center">Malicious URL detection using deep learning methods</h3>

  <p align="center">
    A simple automation framework for Natural Language Processing neural networks training and evaluation.
    Demonstrated on malicious URL classification.
    <br />
    <br />
    <a href="https://github.com/mlorinc/malicious-url-classifier#license">License</a>
    ·
    <a href="https://github.com/mlorinc/malicious-url-classifier#usage">Usage</a>
    ·
    <a href="https://github.com/mlorinc/malicious-url-classifier#reproduction">Reproduction steps</a>
    ·
    <a href="https://github.com/mlorinc/malicious-url-classifier/issues">Report Bug</a>
  </p>
</div>

This project was created as part of my Secure Hardware Devices university course,
in which I had to design LSTM, GRU or Bi-LSTM based neural networks. As I was
conducting more experiments, the code base cohesion was getting worse and worse
by each experiment iteration. Eventually, the code base was hard to maintain.
Therefore, this project was created as side product of the course project.

Entire training datasets, trained URL classification models are not part of the repository,
because of their size. Though, model definitions can be found in the `models` module.
Sadly, the module cannot be configured to custom one, so it is better to clone repository
and work with the tool this way.

Furthermore, strings are solely used for classification and before they are fed into
models, they are converted into vector embeddings. The embedding dimension
is fixed at 128, however other parameters can be set in JSON configurations.

The following parameters can be configured:

| Parameter | Description |
| --- | --- |
| batch_size | training batch size, training performance x accuracy tradeoff |
| max_features | maximum number of characters/words to be used by vectorizer |
| split | either character or whitespace |
| standardize | name of the standardizer |
| max_length | maximum length of the string |
| model | model definition which resides in models module |
| stringify | make model accept strings instead vectors |
| output_path | the path where final model will be saved |
| epoch | number of training epochs |
| patience | number of epochs until the training is stopped due to worsening validation loss |

At last, currently only binary classifiers are supported and training values are hardcoded to `url`
and labels to `type`. Though, it is not complicated to modify for multi-class classification.

### Built With

To run project, make sure you have the following libraries installed:

1. Keras
1. Tensorflow
1. Pandas
1. Seaborn

## License
<a name="license" />

Distributed under the Apache-2.0 License. See `LICENSE.txt` for more information.

## Usage
<a name="usage" />

Program features four commands which can be invoked by launching `src/main.py` file.
The most usual flow has the following order:

1. Merge datasets: `src/main.py dataset:fuse data/fused_dataset.csv <datasets...>`
2. Split dataset into ./dataset folder: `src/main.py dataset:split ./dataset data/fused_dataset.csv 0.6 0.2 0.2`
3. Train NN: `src/main.py model:train ./dataset ./configs/some.json`
4. Evaluate NN and save plot to mygallery: `src/main.py model:evaluate ./mygallery dataset <models...>`

## Malicious URL classification result reproduction
<a name="reproduction"/>

In order to reproduce data located in gallery, follow the following actions:

1. Train CNN-RNN hybrids: `src/main.py model:train ./dataset ./configs/model.configs.json`
2. Train RNN: `src/main.py model:train ./dataset ./configs/recurrent.model.configs.json`
3. Train CNN-RNN-Dense hybrids: `src/main.py model:train ./dataset ./configs/combined.model.configs.json`
4. Perform evaluation: `src/main.py model:evaluate ./figures dataset ./models/* ./memory_models/* ./combined_models/*`