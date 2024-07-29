import argparse
import collections
import dataclasses
import enum
import json
import os
import pathlib
import typing
from typing import Optional, TypeAlias

import datasets
import dataclasses_json
import fasttext
import fasttext.util
import lets_plot as lp
import numpy
import pandas as pd
import torch
import tqdm
import transformers
from fasttext.FastText import _FastText
from sklearn.cluster import KMeans
from tokenizers.implementations import BertWordPieceTokenizer

# Sometimes import take a long time to load, so this is an indication for me whether the imports are
# slow or whether the program itself takes a long time to run.
print("Done importing libraries.")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# For creating the graphs. Colors are from the Tailwind CSS color palette.
COLORS = [
    "#4f46e5",  # indigo-600
    "#db2777",  # pink-600
    "#059669",  # emerald-600
    "#0ea5e9",  # sky-500
    "#facc15",  # yellow-400
]

# For creating the grid lines. Colors are also from the Tailwind CSS color palette.
LINE_COLORS = {
    "100": "#f1f5f9",
    "200": "#e2e8f0",
    "300": "#cbd5e1",
    "600": "#475569",
    "700": "#334155",
    "900": "#0f172a",
}


def get_device() -> torch.device:
    """
    Since I am developing this on a Mac, I want to test the model using the MPS backend. However, I
    also want to be able to run this on a GPU. This function returns the correct device for the
    current system.
    """
    match (torch.cuda.is_available(), torch.backends.mps.is_available()):
        case (True, _):
            return torch.device("cuda")
        case (_, True):
            return torch.device("mps")
        case _:
            return torch.device("cpu")


class Task(enum.Enum):
    """
    To get the final model you will have to run multiple tasks, such as preprocessing the data,
    creating a tokenizer, and finally training the model.
    """

    TRAIN = "train"
    TOKENIZE = "tokenize"
    PREPROCESS = "preprocess"
    DEBUG = "debug"
    GET_SCORES = "get-scores"
    PLOT_STATS = "plot-stats"
    PLOT_DATA = "plot-data"
    PLOT_RESULTS = "plot-results"
    PLOT_UMBRELLA_EXAMPLE = "plot-umbrella-example"


def parse_arguments() -> argparse.Namespace:
    """
    A large parser which will parse the arguments for a given task. The tasks are for different
    parts of my thesis, from preprocessing the data to plotting the results of the training.
    """

    parser = argparse.ArgumentParser(description="Train a model")

    subparsers = parser.add_subparsers(
        help="choose which task you want to run", required=True, dest="task"
    )

    train_parser = subparsers.add_parser("train", help="train a model")
    tokenize_parser = subparsers.add_parser("tokenize", help="train a tokenizer")
    preprocess_parser = subparsers.add_parser("preprocess", help="preprocess the data")
    plot_stats_parser = subparsers.add_parser(
        "plot-stats", help="plot the statistics of the training"
    )
    plot_data_parser = subparsers.add_parser(
        "plot-data",
        help="create statistics about the data and create a plot of the token distribution",
    )
    plot_umbrella_example_parser = subparsers.add_parser(
        "plot-umbrella-example", help="plot the umbrella example"
    )
    _plot_results_parser = subparsers.add_parser(
        "plot-results", help="plot the results of training the model (e.g. the loss)"
    )
    get_scores_parser = subparsers.add_parser(
        "get-scores", help="get the scores of the models and organize them per task in a table"
    )
    _debug_parser = subparsers.add_parser("debug", help="runs the code in the debug case")

    shared_defaults = {
        "vocab_size": 30522,
        "batch_size": 256,
    }

    # Arguments for training the model

    train_parser.add_argument("--train-dir", type=str, required=True)
    train_parser.add_argument("--dev-dir", type=str, default=None)
    train_parser.add_argument("--output-dir", type=str, required=True)
    train_parser.add_argument("--tokenizer", type=str, required=True)
    train_parser.add_argument("--amount-datasets", type=int, default=5)

    train_parser.add_argument("--vocab-size", type=int, default=shared_defaults["vocab_size"])
    train_parser.add_argument("--batch-size", type=int, default=shared_defaults["batch_size"])
    train_parser.add_argument("--hidden-size", type=int, default=768)
    train_parser.add_argument("--gradient-accumulation", type=int, default=1)
    train_parser.add_argument("--num-epochs", type=int, default=3)
    train_parser.add_argument("--learning-rate", type=float, default=5e-5)

    train_parser.add_argument("--default-context-size", type=int, default=32)
    train_parser.add_argument(
        "--use-increased-context-size",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Increase the context size of the model at the end of training.",
    )
    train_parser.add_argument("--increased-context-size", type=int, default=128)
    train_parser.add_argument("--num-epochs-increased-context-size", type=int, default=1)
    train_parser.add_argument("--gradient-accumulation-increased-context-size", type=int, default=4)
    train_parser.add_argument("--batch-size-increased-context-size", type=int, default=64)

    train_parser.add_argument(
        "--baseline",
        action=argparse.BooleanOptionalAction,
        help="The model will be trained on the original dataset, without swapping datasets.",
    )

    train_parser.add_argument(
        "--gradual-change-steps",
        type=int,
        default=1,
        help="Gradually change the dataset to the new dataset. So instead of swapping the dataset, "
        "it will be trained on the new and old dataset. What the maximum of combined datasets will "
        "be. This cannot be higher than the amount of datasets. When this is 2 and the amount of "
        "datasets is 3, the gradual change will look like this:"
        " [1], [1, 2], [2, 3], [3]",
    )

    # Arguments for creating the tokenizer

    tokenize_parser.add_argument("--train-dir", type=str, required=True)
    tokenize_parser.add_argument("--output-dir", type=str, required=True)
    tokenize_parser.add_argument("--vocab-size", type=int, default=shared_defaults["vocab_size"])
    tokenize_parser.add_argument("--min-frequency", type=int, default=2)
    tokenize_parser.add_argument("--batch-size", type=int, default=shared_defaults["batch_size"])

    # Arguments for preprocessing the data

    preprocess_parser.add_argument("--input-dir", type=str, required=True)
    preprocess_parser.add_argument("--output-dir", type=str, required=True)

    preprocess_parser.add_argument(
        "--most-common",
        type=int,
        default=1000,
        help="During preprocessing the vocabulary size is reduced by coupling words together with "
        "similar meaning. However, you probably don't want this to happen to frequently used "
        "words.",
    )
    preprocess_parser.add_argument(
        "--clusters",
        type=int,
        default=1000,
        help="The amount of words clustered words. Words within a cluster get replaced by the most "
        "frequent word in that cluster, which will represent that cluster.",
    )

    # Arguments for plotting the statistics of the training

    plot_stats_parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="given folder names it will look for stats.csv files in those folders",
    )

    plot_stats_parser.add_argument("--output", type=str, required=True)

    # Arguments for plotting the data used for training

    plot_data_parser.add_argument("--input-dir", type=str, required=True)
    plot_data_parser.add_argument("--output", type=str, required=True)

    # Arguments for plotting the umbrella example

    plot_umbrella_example_parser.add_argument("--input", type=str, required=True)
    plot_umbrella_example_parser.add_argument("--output", type=str, required=True)

    # Arguments for getting the results of the training

    get_scores_parser.add_argument("--input-dir", required=True)
    get_scores_parser.add_argument("--output", required=True)
    get_scores_parser.add_argument(
        "--without-names",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Don't include the names of the tests in the output.",
    )

    # Parsing the arguments

    return parser.parse_args()


def load_dataset(
    train_paths: list[str], dev_paths: Optional[list[str]] = None
) -> datasets.DatasetDict:
    paths = {"train": train_paths}

    if dev_paths is not None:
        paths.update({"development": dev_paths})

    dataset = datasets.load_dataset("text", data_files=paths)
    return dataset  # type:ignore


def generate_sequence(amount_items, max_length):
    """
    For the gradual change of the dataset, this function will generate the sequence of datasets to
    train the model on.
    """
    result = []
    for i in range(-max_length + 2, amount_items + 1):
        new = [j for j in range(i, i + max_length) if 0 < j < amount_items + 1]
        result.append(new)
    return result


@dataclasses.dataclass
class TrainConfig:
    """ """

    train_dir: str
    dev_dir: Optional[str]
    output_dir: str
    tokenizer: str
    amount_datasets: int
    gradual_change_steps: int

    vocab_size: int
    batch_size: int
    hidden_size: int
    gradient_accumulation: int
    num_epochs: int
    learning_rate: float

    default_context_size: int
    use_increased_context_size: bool
    increased_context_size: int
    num_epochs_increased_context_size: int
    gradient_accumulation_increased_context_size: int
    batch_size_increased_context_size: int

    baseline: bool

    device: torch.device = get_device()


def create_output_dir_model(
    config: TrainConfig, use_increased_context_size: bool, current_steps: Optional[list[int]] = None
) -> str:
    """
    In the given output directory multiple new directories will be created. Based on the arguments
    those directories can be for steps, increased context size, or the baseline model.
    """
    if use_increased_context_size:
        return os.path.join(config.output_dir, "increased-context-size")
    elif config.baseline:
        return os.path.join(config.output_dir, "baseline")
    elif current_steps is not None:
        return os.path.join(
            config.output_dir, f"step-{'_'.join([str(step) for step in current_steps])}"
        )

    return "model"  # will probably never be used


def create_trainer(
    config: TrainConfig,
    dataset: datasets.DatasetDict,
    tokenizer: transformers.PreTrainedTokenizerBase,
    gradient_accumulation: int,
    batch_size: int,
    num_epochs: int,
    model: transformers.PreTrainedModel,
    use_increased_context_size: bool = False,
    current_steps: Optional[list[int]] = None,
) -> transformers.Trainer:
    """
    With the config created by the argparser and the default values create the trainer for the
    model.
    """
    output_dir = create_output_dir_model(config, use_increased_context_size, current_steps)

    return transformers.Trainer(
        args=transformers.TrainingArguments(
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=config.learning_rate,
            num_train_epochs=num_epochs,
            warmup_steps=500,
            output_dir=output_dir,
            overwrite_output_dir=True,
            save_total_limit=1,
            per_device_eval_batch_size=batch_size,
            per_device_train_batch_size=batch_size,
            eval_strategy="steps",
            eval_steps=1000,
            torch_compile=True,
        ),
        train_dataset=dataset["train"],
        eval_dataset=dataset["development"] if "development" in dataset else None,
        tokenizer=tokenizer,
        model=model,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15,
        ),
    )


def group_texts(examples, expanded_inputs_length):
    """
    This function will group the texts in the dataset into chunks of the given length. With this
    function the model will train on different context sizes.
    """
    # Concatenate all texts.
    try:
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    except TypeError:
        print(examples)
        return

    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this
    # drop, you can customize this part to your needs.
    if total_length >= expanded_inputs_length:
        total_length = (total_length // expanded_inputs_length) * expanded_inputs_length
    # Split by chunks of max_len.

    result = {
        k: [
            t[i : i + expanded_inputs_length]
            for i in range(0, total_length, expanded_inputs_length)
        ]
        for k, t in concatenated_examples.items()
    }

    return result


def create_dataset_training(
    dataset: datasets.DatasetDict,
    tokenizer: transformers.BertTokenizerFast,
    config: TrainConfig,
    context_size: int,
) -> datasets.DatasetDict:
    """
    Additional processing of the dataset is done, so afterwards the dataset can be used for training
    the model.
    """
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples["text"], return_special_tokens_mask=True),
        batched=True,
        batch_size=config.batch_size,
        remove_columns=["text"],
    )

    return tokenized_dataset.map(
        lambda examples: group_texts(examples, context_size),
        batched=True,
        batch_size=config.batch_size,
    )


def get_paths_dataset(
    config: TrainConfig, current_step: Optional[int] = None
) -> tuple[list[str], Optional[list[str]]]:
    """
    Get the paths of the training and development datasets. If the model is a baseline model, the
    training dataset will be the input directory, otherwise the training dataset will be the
    directory of the input directory, which should have subdirectories for each step.
    """
    path_train_dir = (
        config.train_dir
        if config.baseline
        else os.path.join(config.train_dir, f"step-{current_step}")
    )
    paths_train = [str(path) for path in pathlib.Path(path_train_dir).glob("*.train")]
    paths_dev = (
        [str(path) for path in pathlib.Path(config.dev_dir).glob("*.dev")]
        if config.dev_dir
        else None
    )
    return paths_train, paths_dev


def load_dataset_training(config: TrainConfig, current_steps: list[int]) -> datasets.DatasetDict:
    """
    Create a dataset dictionary with the training and development datasets. The training dataset
    will be concatenated if there are multiple current steps (which happens when training the GDC
    model, which is trained on multiple datasets at the same time).
    """
    paths_train, paths_dev = get_paths_dataset(config, current_steps[0])

    if len(current_steps) == 1:
        return load_dataset(paths_train, paths_dev)

    train_datasets: list[datasets.Dataset] = []
    for step in current_steps[1:]:
        paths_train_inner, _ = get_paths_dataset(config, step)
        dataset = load_dataset(paths_train_inner)
        train_datasets.append(dataset["train"])  # type: ignore

    main_dataset = load_dataset(paths_train, paths_dev)
    train_datasets.append(main_dataset["train"])  # type: ignore
    new = datasets.concatenate_datasets(train_datasets)

    return datasets.DatasetDict({"train": new, "development": main_dataset["development"]})


def train_and_save_model(
    trainer: transformers.Trainer,
    config: TrainConfig,
    use_increased_context_size: bool = False,
    current_steps: Optional[list[int]] = None,
) -> transformers.PreTrainedModel:
    """
    Start the training process, and save the model and the stats in the output directory.
    """
    output = create_output_dir_model(config, use_increased_context_size, current_steps)
    stats_path = os.path.join(output, "stats.csv")

    trainer.train()
    trainer.save_model(output)

    pd.DataFrame(trainer.state.log_history).to_csv(stats_path)

    return trainer.model  # type: ignore


def train(config: TrainConfig):
    """
    Train the model from scratch. Given the type of model, the model will be trained as baseline
    model, DC model or as GDC model.
    """
    tokenizer = transformers.BertTokenizerFast.from_pretrained(config.tokenizer)

    model = transformers.BertForMaskedLM(
        transformers.BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )
    )

    if config.baseline:
        train_paths, dev_paths = get_paths_dataset(config)
        dataset = load_dataset(train_paths, dev_paths)
        dataset_training = create_dataset_training(
            dataset, tokenizer, config, config.default_context_size
        )
        trainer = create_trainer(
            config,
            dataset_training,
            tokenizer,
            config.gradient_accumulation,
            config.batch_size,
            config.num_epochs,
            model,
        )
        model = train_and_save_model(trainer, config)
    else:
        steps = generate_sequence(config.amount_datasets, config.gradual_change_steps)
        for step in steps:
            dataset = load_dataset_training(config, step)
            dataset_training = create_dataset_training(
                dataset, tokenizer, config, config.default_context_size
            )
            trainer = create_trainer(
                config,
                dataset_training,
                tokenizer,
                config.gradient_accumulation,
                config.batch_size,
                config.num_epochs,
                model,
                current_steps=step,
            )
            model = train_and_save_model(trainer, config, current_steps=step)

    if config.use_increased_context_size:
        train_paths, dev_paths = get_paths_dataset(
            config, current_step=None if config.baseline else config.amount_datasets
        )
        dataset = load_dataset(train_paths, dev_paths)
        dataset_training = create_dataset_training(
            dataset, tokenizer, config, config.increased_context_size
        )
        trainer = create_trainer(
            config,
            dataset_training,
            tokenizer,
            config.gradient_accumulation_increased_context_size,
            config.batch_size_increased_context_size,
            config.num_epochs_increased_context_size,
            model,
            use_increased_context_size=True,
        )
        _ = train_and_save_model(trainer, config, use_increased_context_size=True)


@dataclasses.dataclass
class TokenizeConfig:
    train_dir: str
    output_dir: str
    vocab_size: int
    min_frequency: int
    batch_size: int


def create_tokenizer(paths: list[str], config: TokenizeConfig) -> BertWordPieceTokenizer:
    tokenizer = BertWordPieceTokenizer(lowercase=True)

    tokenizer.train(
        paths,
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
    )

    return tokenizer


def tokenize(config: TokenizeConfig):
    """
    Train the bert word piece tokenizer and save it to the output directory.
    """
    paths = [str(path) for path in pathlib.Path(config.train_dir).glob("*.train")]
    tokenizer = create_tokenizer(paths, config)
    os.makedirs(config.output_dir, exist_ok=True)
    tokenizer.save_model(config.output_dir)


@dataclasses.dataclass
class PreprocessConfig:
    input_dir: str
    output_dir: str
    most_common: int
    clusters: int


FastTextMap: TypeAlias = dict[str, numpy.ndarray]
ClusterRepresentation: TypeAlias = dict[int, str]
ClusterMap: TypeAlias = dict[str, int]
Tokens: TypeAlias = list[str]
Types: TypeAlias = list[str]
TypesOccurrences: TypeAlias = dict[str, int]


def create_word_embedding_map(
    ft: _FastText, tokens: Tokens, config: PreprocessConfig
) -> tuple[FastTextMap, Types, Types]:
    """
    Get the necessary information to preprocess the data. Get the embeddings and map them to the
    words in the dataset. Also get the types that are present in both fastText and the dataset and
    are not used in the most common words. And return the most common words.
    """
    ft_map = {}

    occurrences = collections.Counter(tokens)
    most_common_types = [token for token, _ in occurrences.most_common(config.most_common)]

    types = list(set(tokens).intersection(set(ft.get_words())).difference(set(most_common_types)))

    for word in tqdm.tqdm(types, desc="mapping fasttext words"):
        ft_map.update({str(word): ft.get_word_vector(word)})
    return ft_map, types, most_common_types


def create_cluster_model(ft_map: FastTextMap, config: PreprocessConfig) -> KMeans:
    model = KMeans(n_clusters=config.clusters).fit(list(ft_map.values()))
    return model  # type: ignore


def get_cluster_representations(
    model: KMeans,
    types: Types,
    tokens: Tokens,
    config: PreprocessConfig,
) -> tuple[ClusterRepresentation, ClusterMap]:
    """
    For each cluster get the most frequent word in that cluster, which will be used as the
    representation or umbrella term for that cluster.
    """
    representation = {}
    cluster_map = {
        type_: int(cluster_id)
        for cluster_id, type_ in zip(model.labels_.tolist(), types)  # type: ignore
    }

    for token, _ in collections.Counter(tokens).most_common():
        cluster_id = cluster_map.get(token)
        if cluster_id not in representation:
            representation.update({cluster_id: token})
        if len(representation) == config.clusters:
            break

    return representation, cluster_map


def preprocess_item(
    item: str,
    representation: ClusterRepresentation,
    cluster_map: ClusterMap,
    types: Types,
    most_common_types: Types,
) -> str:
    """
    This function does the preprocessing for a single entry in the dataset.
    """
    from nltk.tokenize import word_tokenize

    tokens = word_tokenize(item)
    simplified = []

    for token in tokens:
        if token in most_common_types or token not in types or token not in cluster_map:
            simplified.append(token)
            continue

        overarching_token = representation.get(cluster_map.get(token))  # type:ignore

        if overarching_token is None:
            simplified.append(token)
            continue

        simplified.append(overarching_token)

    return " ".join(simplified)


def preprocess(config: PreprocessConfig):
    """
    Preprocess the original dataset from the shared task, by clustering the word embeddings of the
    words in the dataset. This will decrease the vocabulary size of the dataset, and make it easier
    to train a model on the dataset.
    """
    from nltk.tokenize import word_tokenize

    paths = [str(path) for path in pathlib.Path(config.input_dir).glob("*.train")]
    tokens = [
        token
        for path in paths
        for line in open(path, "r").readlines()
        for token in word_tokenize(line)
    ]
    print("amount of tokens:", len(tokens))
    print("loading fasttext model")
    filename = fasttext.util.download_model("en", if_exists="ignore")
    ft = fasttext.load_model(filename)
    print("creating word embedding map")
    ft_map, types, most_common_types = create_word_embedding_map(ft, tokens, config)
    print("creating cluster model")
    model = create_cluster_model(ft_map, config)
    print("creating cluster representations")
    representations, cluster_map = get_cluster_representations(model, types, tokens, config)

    output_paths = [os.path.join(config.output_dir, os.path.basename(path)) for path in paths]

    os.makedirs(config.output_dir, exist_ok=True)

    for input_path, output_path in zip(paths, output_paths):
        input_file, output_file = open(input_path, "r"), open(output_path, "w")
        for line in tqdm.tqdm(input_file.readlines()):
            output_file.write(
                preprocess_item(line, representations, cluster_map, types, most_common_types) + "\n"
            )
        input_file.close()
        output_file.close()

    with open(os.path.join(config.output_dir, "cluster_map.json"), "w") as f:
        json.dump(cluster_map, f)

    with open(os.path.join(config.output_dir, "cluster_representations.json"), "w") as f:
        json.dump(representations, f)

    with open(os.path.join(config.output_dir, "occurrences.json"), "w") as f:
        json.dump(dict(collections.Counter(tokens)), f)


def basic_theme() -> lp.Dict:
    """
    Provide the settings for a custom theme that can be used with `lets-plot`.
    """
    return lp.theme(
        text=lp.element_text(family="Inter", color=LINE_COLORS["900"], size=18),
        axis_text=lp.element_text(size=14, color=LINE_COLORS["600"]),
        legend_text=lp.element_text(size=14, color=LINE_COLORS["900"]),
        panel_grid_major_y=lp.element_line(color=LINE_COLORS["300"]),
        panel_grid_major_x=lp.element_line(color=LINE_COLORS["300"]),
    ) + lp.scale_color_manual(values=COLORS)  # type: ignore


@dataclasses.dataclass
class PlotStatsConfig:
    input_dirs: list[str]
    output: str


def plot_stats(config: PlotStatsConfig):
    """
    Given the paths to the directories in which the models and there stats.csv files are stored,
    this function will combine the stats.csv files for models where parts of the models are saved in
    different subdirectories (for example, the DC model will have multiple subdirectories, since a
    new directory is created each time the model switches dataset). Then from the combined stats,
    the following plots will be created:

    - A plot of the training loss for each model
    - A plot of the validation loss for each model
    """

    df = pd.DataFrame()

    for input_dir in config.input_dirs:
        name = pathlib.Path(input_dir).parts[-1]
        paths = [str(path) for path in pathlib.Path(input_dir).glob("**/stats.csv")]

        increased_context_size_path = None
        other_paths = []

        for path in paths:
            if "increased-context-size" in path:
                increased_context_size_path = path
            else:
                other_paths.append(path)

        other_paths = sorted(other_paths)
        paths = (
            other_paths + [increased_context_size_path]
            if increased_context_size_path
            else other_paths
        )

        df_model = None

        for index, path in enumerate(paths):
            df_round = pd.read_csv(path)
            df_round["increased_context_size"] = "increased-context-size" in path

            if index == 0:
                df_model = df_round
            else:
                previous_highest_step = df_model["step"].max()  # type: ignore
                df_round["step"] = df_round["step"] + previous_highest_step
                df_model = pd.concat([df_model, df_round], axis=0)  # type: ignore

        df_model["name"] = name  # type: ignore
        df = pd.concat([df, df_model], axis=0)  # type: ignore

    y_lims = [
        min(df["eval_loss"].min(), df["loss"].min()),
        max(df["eval_loss"].max(), df["loss"].max()),
    ]

    plot = (
        lp.ggplot(df, lp.aes(x="step", y="eval_loss", color="name"))
        + lp.geom_line()
        + lp.xlab("Step")
        + lp.ylab("Evaluation Loss")
        + lp.lims(y=y_lims, x=[None, None])
        + basic_theme()
        + lp.theme(
            legend_position="bottom",
            legend_title=lp.element_blank(),
            panel_grid_minor_y=lp.element_line(color=LINE_COLORS["200"]),
        )
    )

    lp.ggsave(plot, filename="eval_loss.png", path=config.output, scale=4)  # type: ignore

    plot = (
        lp.ggplot(df, lp.aes(x="step", y="loss", color="name"))
        + lp.geom_line()
        + lp.xlab("Step")
        + lp.ylab("Training Loss")
        + lp.lims(y=y_lims, x=[None, None])
        + basic_theme()
        + lp.theme(
            legend_position="bottom",
            legend_title=lp.element_blank(),
            panel_grid_minor_y=lp.element_line(color=LINE_COLORS["200"]),
        )
    )

    lp.ggsave(plot, filename="training_loss.png", path=config.output, scale=4)  # type: ignore


@dataclasses.dataclass
class PlotDataConfig:
    input_dir: str
    output: str


def plot_data(config: PlotDataConfig):
    """
    Get statistics about the data and make a plot of the frequency distribution of the tokens.
    """
    from nltk.tokenize import sent_tokenize, word_tokenize

    os.makedirs(config.output, exist_ok=True)
    paths = [
        str(path)
        for path in pathlib.Path(config.input_dir).glob("*.*")
        if path.suffix in [".train", ".dev"]
    ]
    lines = [line for path in paths for line in open(path, "r").readlines()]

    tokens = [token for line in lines for token in word_tokenize(line)]
    sentences = [sentence for line in lines for sentence in sent_tokenize(line)]
    occurrences = collections.Counter(tokens)

    stats = {
        "amount_tokens": len(tokens),
        "amount_sentences": len(sentences),
        "amount_types": len(occurrences),
        "type_token_ratio": len(occurrences) / len(tokens),
        "average_sentence_length": len(tokens) / len(sentences),
    }

    df_token_distribution = pd.DataFrame(
        occurrences.items(),
        columns=["token", "occurrences"],  # type: ignore
    ).sort_values("occurrences", ascending=False)

    df_token_distribution["id"] = numpy.arange(0, len(df_token_distribution))

    plot = (
        lp.ggplot(df_token_distribution, lp.aes(x="id", y="occurrences"))
        + lp.geom_point(size=0.75)
        + lp.scale_y_log10()
        + lp.lims(x=[0, 225000], y=[0, 825000])
        + basic_theme()
        + lp.xlab("Types")
        + lp.ylab("Occurrences")
        + lp.theme(panel_inset=[0, 0, 8], geom=lp.element_geom(pen=COLORS[0]))
    )

    lp.ggsave(plot, filename="token_distribution.png", scale=4, path=config.output)  # type: ignore

    df_stats = pd.DataFrame(stats.items(), columns=["stat", "value"])  # type: ignore
    df_stats.to_csv(os.path.join(config.output, "stats.csv"))


@dataclasses.dataclass
class GetScoresConfig:
    input_dir: str
    output: str
    without_names: bool = False


GLUE_TESTS = [
    "cola",
    "sst2",
    "mrpc",
    "qqp",
    "mnli",
    "mnli-mm",
    "qnli",
    "rte",
    "boolq",
    "multirc",
    "wsc",
]
MSGS_TESTS = [
    "main_verb_control",
    "control_raising_control",
    "syntactic_category_control",
    "lexical_content_the_control",
    "relative_position_control",
    "main_verb_lexical_content_the",
    "main_verb_relative_token_position",
    "syntactic_category_lexical_content_the",
    "syntactic_category_relative_position",
    "control_raising_lexical_content_the",
    "control_raising_relative_token_position",
]


def format_score(score: float) -> str:
    percentage = score * 100
    return "{:.1f}".format(percentage)


def get_scores(config: GetScoresConfig) -> None:
    """
    With the input directory, the model will find all the eval_results.json files in which the
    scores from the model on the evaluation tasks are stored. Since there are many subtasks, doing
    this manually would be very time-consuming.

    The results are grouped by task. For the BLiMP and GLUE (except mrpc and qqp) task, the accuracy
    is stored, and for the MRPC and QQP subtask, the F1 score is stored. For the MSGS task, the
    Matthews correlation coefficient is stored.
    """
    paths = pathlib.Path(config.input_dir).glob("**/eval_results.json")

    scores = {"glue": [], "msgs": [], "blimp": []}

    for path in paths:
        name = path.parts[-2]

        with open(path, "r") as file:
            data = json.load(file)

            if name in GLUE_TESTS:
                index = GLUE_TESTS.index(name)
                score = data["eval_f1"] if name in ["mrpc", "qqp"] else data["eval_accuracy"]
                scores["glue"].append(
                    {"name": name.replace("_", " "), "score": score, "index": index}
                )
            elif name in MSGS_TESTS:
                index = MSGS_TESTS.index(name)
                score = data["eval_mcc"]
                scores["msgs"].append(
                    {"name": name.replace("_", " "), "score": score, "index": index}
                )
            else:
                score = data["eval_accuracy"]
                scores["blimp"].append({"name": name.replace("_", " "), "score": score})

    os.makedirs(config.output, exist_ok=True)

    for key, value in scores.items():
        df = pd.DataFrame(value)

        if key != "blimp":
            df = df.sort_values("index", ascending=True)
            df = df.drop(columns=["index"])
        else:
            df = df.sort_values("name", ascending=True)

        if config.without_names:
            df = df.drop(columns=["name"])

        df["score"] = df["score"].apply(format_score)

        df.to_csv(os.path.join(config.output, f"{key}.csv"), index=False)


@dataclasses.dataclass
class PlotUmbrellaExampleConfig:
    input: str
    output: str


@dataclasses.dataclass
@dataclasses_json.dataclass_json
class ParentWordPair:
    parent: str
    word: str


def add_edges(
    words: list[ParentWordPair], acc: typing.Any = [], previous_parent=None
) -> typing.Any:
    """
    ...
    """
    head, *tail = words

    current_parent = head.parent

    if current_parent == previous_parent:
        pass

    if tail == []:
        return acc
    else:
        return add_edges(tail, "_", current_parent)


def plot_umbrella_example(config: PlotUmbrellaExampleConfig) -> None:
    """
    Given a file with the parent word pairs, the function will create a graph with the parent word
    pairs as nodes. This is to illustrate the idea of using umbrella terms for simpler datasets.
    """
    return None


def main():
    arguments = vars(parse_arguments())
    task = Task(arguments.pop("task"))

    match task:
        case Task.TRAIN:
            train_config = TrainConfig(**arguments)
            train(train_config)
        case Task.TOKENIZE:
            tokenize_config = TokenizeConfig(**arguments)
            tokenize(tokenize_config)
        case Task.PREPROCESS:
            preprocess_config = PreprocessConfig(**arguments)
            preprocess(preprocess_config)
        case Task.PLOT_STATS:
            plot_stats_config = PlotStatsConfig(**arguments)
            plot_stats(plot_stats_config)
        case Task.PLOT_DATA:
            plot_data_config = PlotDataConfig(**arguments)
            plot_data(plot_data_config)
        case Task.PLOT_UMBRELLA_EXAMPLE:
            plot_umbrella_example_config = PlotUmbrellaExampleConfig(**arguments)
            plot_umbrella_example(plot_umbrella_example_config)
        case Task.GET_SCORES:
            get_scores_config = GetScoresConfig(**arguments)
            get_scores(get_scores_config)
        case Task.DEBUG:
            print("for testing functions or parts of the code for other tasks")
            print("debug")


if __name__ == "__main__":
    main()
