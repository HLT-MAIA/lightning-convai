# -*- coding: utf-8 -*-
r""" 
GPT2 Language Model
==================
    GPT2 Language Model implementing the PyTorch Lightning interface that can be used to train a taskmaster Chatbot.
"""
import os
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from transformers import AdamW, GPT2DoubleHeadsModel
from transformers.modeling_gpt2 import GPT2DoubleHeadsModelOutput

import pytorch_lightning as pl
from models.tokenizer import Tokenizer
from pytorch_lightning.metrics.functional import accuracy
from utils import Config
from models.base_model import AssistantBase, AssitantModelOutput


class AssistantGPT2(AssistantBase):
    """GPT2 Model implementing the PyTorch Lightning interface that can be used to
        train a taskmaster Chatbot.

    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams = hparams
        self.hparams.model = "AssistantGPT2"
        self.model = GPT2DoubleHeadsModel.from_pretrained(
            self.hparams.pretrained_model, output_hidden_states=True
        )

        self.tokenizer = Tokenizer(self.hparams.pretrained_model)
        # Resize embeddings to include the added tokens
        self.model.resize_token_embeddings(self.tokenizer.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        mc_token_ids: torch.Tensor,
        lm_labels: torch.Tensor = None,
        mc_labels: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
    ) -> GPT2DoubleHeadsModelOutput:
        gpt2_output = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            mc_token_ids=mc_token_ids,
            mc_labels=mc_labels,
            labels=lm_labels,
            return_dict=True,
        )
        return AssitantModelOutput(
            lm_loss=gpt2_output.loss,
            mc_loss=gpt2_output.mc_loss,
            lm_logits=gpt2_output.logits,
            mc_logits=gpt2_output.mc_logits,
            past_key_values=gpt2_output.past_key_values,
            hidden_states=gpt2_output.hidden_states,
            attentions=gpt2_output.attentions,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        num_return_sequences: Optional[int] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Redirects to GPT2 generate function:
        - https://huggingface.co/transformers/main_classes/model.html#transformers.generation_utils.GenerationMixin.generate
        """
        return self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.eos_index,
            num_return_sequences=num_return_sequences,
            token_type_ids=token_type_ids,
        )
