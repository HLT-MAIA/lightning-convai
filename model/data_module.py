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
        domain: List[int],
        history: List[List[int]],
        reply: List[int] = [],
        lm_labels: bool = False,
    ) -> Dict[str, List[int]]:
        """Builds a model input.

        :param domain: Domain sentence tokenized.
        :param history: List of history sentences tokenizes.
        :param reply: Tokenized answer.
        :param lm_labels: Flag to build LM labels for ground-truth replies.

        :return: Dictionary with model inputs.
        """
        bos, eos, user, assistant = (
            tokenizer.bos_index,
            tokenizer.eos_index,
            tokenizer.user_index,
            tokenizer.assistant_index,
        )

        sequence = (
            [[bos] + domain]  # concats all domain sentences
            + history  # concats history
            + [[assistant] + reply + [eos]]  # concats reply
        )
        instance = {
            "input_ids": list(chain(*sequence)),
            "token_type_ids": [
                user if s[0] == bos or s[0] == user else assistant
                for s in sequence
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
        dataset_path: str,
        data_folder: str = "data/",
    ):
        """ Reads dataset from data/ folder
        :param dataset_path: Path to a json file containing the train and validation dataset.
        :param data_folder: Folder used to store data.

        :return: Returns a dictionary with the training and validation data.
        """
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
        taskmaster = self._get_dataset(self.hparams.dataset_path)

        click.secho("Building inputs and labels.", fg="yellow")
        datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
        for dataset_name, dataset in taskmaster.items():

            num_candidates = len(dataset[0]["utterances"][0]["candidates"])
            if self.hparams.num_candidates > 0 and dataset_name == "train":
                num_candidates = min(self.hparams.num_candidates, num_candidates)

            for dialog in dataset:
                domain = dialog["domain"].copy()

                for l, utterance in enumerate(dialog["utterances"]):
                    history = utterance["history"][
                        -(2 * self.hparams.max_history + 1) :
                    ]

                    for j, candidate in enumerate(
                        utterance["candidates"][-num_candidates:]
                    ):
                        lm_labels = bool(j == num_candidates - 1)
                        instance = self.build_input(
                            self.tokenizer, domain, history, candidate, lm_labels
                        )
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                            
                    datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates

        click.secho("Padding inputs and building tensors.", fg="yellow")
        tensor_datasets = {"train": [], "valid": []}
        for dataset_name, dataset in datasets.items():
            dataset = self.pad_dataset(dataset, padding=self.tokenizer.pad_index)

            for input_name in MODEL_INPUTS:
                tensor =  dataset[input_name])
                    
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
