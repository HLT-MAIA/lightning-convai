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
            + history         # concats history
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
    
    def setup(self, stage) -> None:
        """Data preparation function called before training by Lightning.
        Equivalent to the prepare_data in previous Lightning Versions"""
        
        with open(self.hparams.train_data) as json_file:
            self.train_dataset = json.load(json_file)

        with open(self.hparams.valid_data) as json_file:
            self.valid_dataset = json.load(json_file)
            
    def build_training_batch(self, data):
        return self.prepare_batch(data, train=True)

    def build_validation_batch(self, data):
        return self.prepare_batch(data, train=False)

    def prepare_batch(self, data, train=True):
        data = self._tokenize(data)
        num_candidates = min([len(s["candidates"]) for s in data])
        if self.hparams.num_candidates > 0 and train:
            num_candidates = min(self.hparams.num_candidates, num_candidates)

        batch = {k: [] for k in MODEL_INPUTS}
        for utterance in data:
            history = utterance["history"][
                -(2 * self.hparams.max_history + 1):
            ]
            for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                lm_labels = bool(j == num_candidates - 1)
                instance = self.build_input(
                    self.tokenizer, utterance["domain"], history, candidate, lm_labels
                )
                for input_name, input_array in instance.items():
                    batch[input_name].append(input_array)

            batch["mc_labels"].append(num_candidates - 1)
            batch["n_candidates"] = num_candidates

        batch = self.pad_dataset(batch, padding=self.tokenizer.pad_index)
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(batch[input_name])
            # MC labels contain the labels within the batch!
            # Thats why we have to split the data according to those batches.
            if input_name != "mc_labels":
                tensor = tensor.view(
                    (-1, batch["n_candidates"]) + tensor.shape[1:]
                )
            batch[input_name] = tensor

        del batch["n_candidates"]
        return tuple(batch.values())

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """

        def build_training_batch(sample):
            return self.prepare_batch(sample, train=True)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
            collate_fn=build_training_batch
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """

        def build_validation_batch(sample):
            return self.prepare_batch(sample, train=False)

        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
            collate_fn=self.build_validation_batch
        )
