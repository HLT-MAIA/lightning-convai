# -*- coding: utf-8 -*-
r""" 
DataModule
==========
    The DataModule encapsulates all the steps needed to process data:
    - Download / tokenize
    - Save to disk.
    - Apply transforms (tokenize, pad, batch creation, etc…).
    - Load inside Dataset.
    - Wrap inside a DataLoader.

The most important function to understand inside the DataModule is the `build_input` which 
is responsible by building the inputs that will be used to train the PersonaGPT2 Model.

This function receives a tokenized `persona`, `history` and `reply`, concatenates everything
and builds the language model targets. It also keeps track of the possition of the token used
to represent the entire sequence (the last one).

Example:
>>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
>>> DataModule.build_input(
    tokenizer=tokenizer,
    persona=[[72, 588], [1820, 318]],
    history=[[5303, 1804], [316, 20023]],
    reply=[276, 764],
    lm_labels=False
)
{'input_ids': [50258, 72, 588, 1820, 318, 50260, 5303, 1804, 50261, 316, 20023, 50260, 
276, 764, 50258], 'token_type_ids': [50260, 50260, 50260, 50260, 50260, 50261, 50261, 
50261, 50260, 50260, 50260, 50261, 50261, 50261, 50261], 'mc_token_ids': 14, 'lm_labels': 
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]}

>>> DataModule.build_input(
    tokenizer=tokenizer,
    persona=[[72, 588], [1820, 318]],
    history=[[5303, 1804], [316, 20023]],
    reply=[276, 764],
    lm_labels=True
)
{'input_ids': [50258, 72, 588, 1820, 318, 50260, 5303, 1804, 50261, 316, 20023, 50260, 
276, 764, 50258],  'token_type_ids': [50260, 50260, 50260, 50260, 50260, 50261, 50261, 
50261, 50260, 50260, 50260, 50261, 50261, 50261, 50261], 'mc_token_ids': 14, 'lm_labels': 
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 276, 764, 50258]}
"""
import hashlib
import json
import multiprocessing
import os
from argparse import Namespace
from collections import defaultdict
from itertools import chain
from typing import Dict, List

import click
import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from model.tokenizer import Tokenizer
from torchnlp.download import download_file_maybe_extract

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]


class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule.

    :param hparams: Namespace with data specific arguments.
    :param tokenizer: Model Tokenizer.

    """

    def __init__(self, hparams: Namespace, tokenizer: Tokenizer):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = tokenizer

    @classmethod
    def build_input(
        cls,
        tokenizer: Tokenizer,
        persona: List[List[int]],
        history: List[List[int]],
        reply: List[int] = [],
        lm_labels: bool = False,
    ) -> Dict[str, List[int]]:
        """Builds a model input.

        :param persona: List of persona sentences tokenized.
        :param history: List of history sentences tokenizes.
        :param reply: Tokenized answer.
        :param lm_labels: Flag to build LM labels for ground-truth replies.

        :return: Dictionary with model inputs.
        """
        bos, eos, speaker1, speaker2 = (
            tokenizer.bos_index,
            tokenizer.eos_index,
            tokenizer.speaker1_index,
            tokenizer.speaker2_index,
        )

        sequence = (
            [[bos] + list(chain(*persona))]  # concats all persona sentences
            + history  # concats history
            + [reply + [eos]]  # concats reply
        )
        sequence = [sequence[0]] + [
            [speaker2 if (len(sequence) - i) % 2 else speaker1] + s
            for i, s in enumerate(sequence[1:])
        ]

        instance = {
            "input_ids": list(chain(*sequence)),
            "token_type_ids": [
                speaker2 if i % 2 else speaker1
                for i, s in enumerate(sequence)
                for _ in s
            ],
        }
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        instance["lm_labels"] = [-100] * len(instance["input_ids"])
        if lm_labels:
            instance["lm_labels"] = (
                ([-100] * sum(len(s) for s in sequence[:-1]))
                + [-100]
                + sequence[-1][1:]
            )
        return instance

    def _tokenize(self, obj):
        if isinstance(obj, str):
            return self.tokenizer.encode(obj)

        if isinstance(obj, dict):
            return dict((k, self._tokenize(o)) for k, o in obj.items())

        return list(self._tokenize(o) for o in obj)

    def _get_dataset(
        self,
        dataset_path: str = "",
        data_folder: str = "data/",
    ):
        """Downloads PersonaChat corpus from S3 if no dataset_path is provided.

        :param dataset_path: Path to a json file containing the train and validation dataset.
        :param data_folder: Folder used to store data.

        :return: Returns a dictionary with the training and validation data.
        """
        if not os.path.isfile(dataset_path):
            click.secho(f"Download dataset from {PERSONACHAT_URL}", fg="yellow")
            dataset_file = download_file_maybe_extract(
                PERSONACHAT_URL,
                directory=data_folder,
                check_files=["personachat_self_original.json"],
            )
            dataset_path = "data/personachat_self_original.json"

        dataset_hash = (
            int(hashlib.sha256(dataset_path.encode("utf-8")).hexdigest(), 16) % 10 ** 8
        )
        # To avoid using cache for different models
        # split(/) for microsoft/DialoGPT-small
        pretrained_model = (
            self.hparams.pretrained_model.split("/")[1]
            if "/" in self.hparams.pretrained_model
            else self.hparams.pretrained_model
        )
        dataset_cache = data_folder + ".dataset_" + str(dataset_hash) + pretrained_model

        if os.path.isfile(dataset_cache):
            click.secho(f"Loading tokenized dataset from cache: {dataset_cache}.")
            dataset = torch.load(dataset_cache)
            personalities = [
                dialog["personality"]
                for dataset in dataset.values()
                for dialog in dataset
            ]
            return dataset
        else:
            dataset_file = dataset_path

        with open(dataset_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        click.secho("Running tokenization: This might take some time!", fg="yellow")
        dataset = self._tokenize(dataset)
        torch.save(dataset, dataset_cache)

        return dataset

    @classmethod
    def pad_dataset(
        cls, dataset: dict, padding: int = 0, padded_inputs: List[str] = PADDED_INPUTS
    ):
        """
        Pad the dataset.
        NOTE: This could be optimized by defining a Dataset class and
        padding at the batch level, but this is simpler.

        :param dataset: Dictionary with sequences to pad.
        :param padding: padding index.
        :param padded_inputs:
        """
        max_l = max(len(x) for x in dataset["input_ids"])
        for name in padded_inputs:
            dataset[name] = [
                x + [padding if name != "lm_labels" else -100] * (max_l - len(x))
                for x in dataset[name]
            ]
        return dataset

    def prepare_data(self):
        """
        Lightning DataModule function that will be used to load/download data,
        build inputs with padding and to store everything as TensorDatasets.
        """
        personachat = self._get_dataset(self.hparams.dataset_path)

        click.secho("Building inputs and labels.", fg="yellow")
        datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
        for dataset_name, dataset in personachat.items():
            num_candidates = len(dataset[0]["utterances"][0]["candidates"])
            if self.hparams.num_candidates > 0 and dataset_name == "train":
                num_candidates = min(self.hparams.num_candidates, num_candidates)

            for dialog in dataset:
                persona = dialog["personality"].copy()

                for _ in range(self.hparams.personality_permutations):

                    for utterance in dialog["utterances"]:
                        history = utterance["history"][
                            -(2 * self.hparams.max_history + 1) :
                        ]

                        for j, candidate in enumerate(
                            utterance["candidates"][-num_candidates:]
                        ):
                            lm_labels = bool(j == num_candidates - 1)
                            instance = self.build_input(
                                self.tokenizer, persona, history, candidate, lm_labels
                            )

                            for input_name, input_array in instance.items():
                                datasets[dataset_name][input_name].append(input_array)

                        datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                        datasets[dataset_name]["n_candidates"] = num_candidates

                    persona = [persona[-1]] + persona[:-1]  # permuted personalities

        click.secho("Padding inputs and building tensors.", fg="yellow")
        tensor_datasets = {"train": [], "valid": []}
        for dataset_name, dataset in datasets.items():
            dataset = self.pad_dataset(dataset, padding=self.tokenizer.pad_index)

            for input_name in MODEL_INPUTS:
                tensor = torch.tensor(dataset[input_name])

                # MC labels contain the labels within the batch!
                # Thats why we have to split the data according to those batches.
                if input_name != "mc_labels":
                    tensor = tensor.view(
                        (-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:]
                    )

                tensor_datasets[dataset_name].append(tensor)

        self.train_dataset = TensorDataset(*tensor_datasets["train"])
        self.valid_dataset = TensorDataset(*tensor_datasets["valid"])
        click.secho(
            "Train dataset (Batch, Candidates, Seq length): {}".format(
                self.train_dataset.tensors[0].shape
            ),
            fg="yellow",
        )
        click.secho(
            "Valid dataset (Batch, Candidates, Seq length): {}".format(
                self.valid_dataset.tensors[0].shape
            ),
            fg="yellow",
        )

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
        )
