# -*- coding: utf-8 -*-
from argparse import Namespace
from itertools import chain
from typing import Dict, List

from models.data_module import BaseDataModule
from models.tokenizer import Tokenizer


class GPT2DataModule(BaseDataModule):
    """PyTorch Lightning DataModule.

    :param hparams: Namespace with data specific arguments.
    :param tokenizer: Model Tokenizer.

    """

    @classmethod
    def model_inputs(cls):
        return ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]

    @classmethod
    def padded_inputs(cls):
        return ["input_ids", "lm_labels", "token_type_ids"]

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

    @classmethod
    def pad_dataset(cls, dataset: dict, padding: int):
        """
        Pad the dataset.
        NOTE: This could be optimized by defining a Dataset class and
        padding at the batch level, but this is simpler.

        :param dataset: Dictionary with sequences to pad.
        :param padding: padding index.
        :param padded_inputs:
        """
        max_l = max(len(x) for x in dataset["input_ids"])
        for name in cls.padded_inputs():
            dataset[name] = [
                x + [padding if name != "lm_labels" else -100] * (max_l - len(x))
                for x in dataset[name]
            ]
        return dataset
