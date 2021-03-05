# -*- coding: utf-8 -*-
r""" 
MT5 Sequence-to-Sequence Model
=============================
    MT5 Sequence-to-Sequence Model implementing the PyTorch Lightning interface that can be used to train a taskmaster Chatbot.
"""
from argparse import Namespace
from typing import Optional

import torch
from models.base_model import AssistantBase, AssitantModelOutput
from models.tokenizer import Tokenizer
from transformers import MT5ForConditionalGeneration
from transformers.modeling_utils import SequenceSummary


class AssistantMT5(AssistantBase):
    """MT5 Model implementing the PyTorch Lightning interface that can be used to
        train a taskmaster Chatbot.

    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams = hparams
        self.hparams.model = "AssistantMT5"
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(
            self.hparams.pretrained_model, output_hidden_states=True
        )
        self.tokenizer = Tokenizer(self.hparams.pretrained_model)
        # Resize embeddings to include the added tokens
        self.mt5.resize_token_embeddings(self.tokenizer.vocab_size)
        self.mt5.config.update(
            {
                "summary_activation": None,
                "summary_first_dropout": 0.1,
                "summary_proj_to_labels": True,
                "summary_type": "cls_index",
                "summary_use_proj": True,
                "num_labels": 1,
            }
        )
        self.multiple_choice_head = SequenceSummary(self.mt5.config)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mc_token_ids: torch.Tensor = None,
        mc_labels: torch.Tensor = None,
        lm_labels: torch.Tensor = None,
    ):
        # Reshape tensors
        encoder_input_sz = encoder_input_ids.size()
        encoder_input_ids = encoder_input_ids.view(-1, encoder_input_sz[-1])

        attn_mask_sz = attention_mask.size()
        attention_mask = attention_mask.view(-1, attn_mask_sz[-1])

        decoder_input_sz = decoder_input_ids.size()
        decoder_input_ids = decoder_input_ids.view(-1, decoder_input_sz[-1])

        labels_sz = lm_labels.size()
        lm_labels = lm_labels.view(-1, labels_sz[-1])

        mt5_output = self.mt5(
            input_ids=encoder_input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=lm_labels,
            return_dict=True,
        )
        # Get decoder hidden states
        decoder_hidden_states = mt5_output.decoder_hidden_states[-1]
        batch_sz, candidates, seq_sz = decoder_input_sz
        hidden_states = decoder_hidden_states.view(
            batch_sz, candidates, seq_sz, self.mt5.model_dim
        )
        # Compute Multiple Choice Scores
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)
        # Compute Multiple Choice Loss
        mc_loss = None
        if mc_labels is not None:
            mc_loss_fct = torch.nn.CrossEntropyLoss()
            mc_loss = mc_loss_fct(mc_logits, mc_labels)

        return AssitantModelOutput(
            lm_loss=mt5_output.loss,
            mc_loss=mc_loss,
            lm_logits=mt5_output.logits,
            mc_logits=mc_logits,
            past_key_values=mt5_output.past_key_values,
            hidden_states=hidden_states,
            attentions=mt5_output.decoder_attentions,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
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
        """Redirects to Hugging Face generate function:
        - https://huggingface.co/transformers/main_classes/model.html#transformers.generation_utils.GenerationMixin.generate
        """
        return self.mt5.generate(
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
        )
