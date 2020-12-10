# -*- coding: utf-8 -*-
from itertools import chain
from typing import Dict, List

from models.tokenizer import Tokenizer
from models.data_module import BaseDataModule


class T5DataModule(BaseDataModule):
    @classmethod
    def model_inputs(cls):
        return [
            "encoder_input_ids",
            "decoder_input_ids",
            "attention_mask",
            "mc_labels",
            "lm_labels",
            "mc_token_ids",
        ]

    @classmethod
    def padded_inputs(cls):
        return ["encoder_input_ids", "decoder_input_ids", "attention_mask", "lm_labels"]

    @classmethod
    def build_input(
        cls,
        tokenizer: Tokenizer,
        domain: List[int],
        history: List[List[int]],
        reply: List[int],
        lm_labels: bool = True,
    ) -> Dict[str, List[int]]:
        """Builds a model input.

        :param domain: Domain sentence tokenized.
        :param history: List of history sentences tokenizes.
        :param reply: Tokenized answer.

        :return: Dictionary with model inputs.
        """
        bos, eos, pad = (tokenizer.bos_index, tokenizer.eos_index, tokenizer.pad_index)
        source = (
            [[bos] + domain]  # concats all domain sentences
            + history
            + [[eos]]  # concats history
        )
        source = list(chain(*source))

        instance = {
            "encoder_input_ids": source,
            "attention_mask": [1 for _ in source],
        }
        instance["decoder_input_ids"] = [pad] + reply + [eos]
        instance["lm_labels"] = [-100] * len(instance["decoder_input_ids"])
        instance["mc_token_ids"] = len(instance["lm_labels"]) - 1
        if lm_labels:
            instance["lm_labels"] = reply + [eos]

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
        max_src = max(len(x) for x in dataset["encoder_input_ids"])
        max_tgt = max(len(x) for x in dataset["decoder_input_ids"])

        for name in cls.padded_inputs():
            if name == "lm_labels":
                dataset[name] = [x + [-100] * (max_tgt - len(x)) for x in dataset[name]]
            elif name == "decoder_input_ids":
                dataset[name] = [
                    x + [padding] * (max_tgt - len(x)) for x in dataset[name]
                ]
            else:
                dataset[name] = [
                    x + [padding] * (max_src - len(x)) for x in dataset[name]
                ]
        return dataset
