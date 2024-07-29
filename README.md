# thesis-master

This is the repository for my master thesis. For my thesis a BERT model was trained on a dataset with limited data. The challange was to create a well-performing model the limited amount of data.

To do so, we used a variant of curriculum learning, where the model was trained on a simpler dataset first, and then moved on to the more complex dataset.

The datasets were simplified by reducing the amount of types in the dataset. We did this by representing infrequently used types by a more frequently used umbrella term.


## Data
The data for training this model is available on the website of the BabyLM shared task. We used the data from last year's shared task. Here is the link to last year's shared task: [BabyLM 2023](https://babylm.github.io/archive_2023.html)

## Files
The repository contains the following files:
- `main.py`: The file with which you can preprocess the data, train a tokenizer and BERT model, and plot statistics about the data and results of the model.
- `requirements.txt`: The file with all the required packages to run the code.
- `.tool-versions`: The file with the versions of programs used in this project. We only used Python 3.11.0. If you have `asdf` installed, you can run `asdf install` to install the correct version of Python.
- `scores/`: The directory with the scores of the model per task and per model.

## Usage
To run the code, you need to have Python 3.11.0 installed. When you have Python 3.11.0 installed, you can install the required packages by running the following command:

```bash
python -m venv .venv # Create a virtual environment
pip install -r requirements.txt # Install the required packages
```

After you have installed the required packages, you can run the code by running the following command:

```bash
python main.py --help
```

This will show which commands are available to run certain tasks. For example, to preprocess the data and create dataset 1, you can run the following command:

```bash
python main.py preprocess --input-dir data/babylm_data/babylm_10M --output-dir data/preprocessed/step-1 --most-common 1000 --clusters 500
```
