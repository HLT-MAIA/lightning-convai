# -*- coding: utf-8 -*-
r""" 
PersonaGPT2 Model
==================
    GPT2 Model implementing the PyTorch Lightning interface that can be used to train a Persona Chatbot.
"""
import os
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from transformers import AdamW, GPT2DoubleHeadsModel
from transformers.modeling_gpt2 import GPT2DoubleHeadsModelOutput

import pytorch_lightning as pl
from model.tokenizer import Tokenizer
from pytorch_lightning.metrics.functional import accuracy
from utils import Config


class PersonaGPT2(pl.LightningModule):
    """GPT2 Model implementing the PyTorch Lightning interface that can be used to
        train a Persona Chatbot.

    :param hparams: ArgumentParser containing the hyperparameters.
    """

    class ModelConfig(Config):
        """The ModelConfig class is used to define Model settings.

        :param pretrained_model: Pretrained GPT2 model to be used.
        :param learning_rate: Learning Rate used during training.
        :param lm_coef: Weight assigned to the LM loss.
        :param mc_coef: Weight assigned to the Multiple-Choice loss.
        :param dataset_path: Path to a json file containing our data.
        :param batch_size: Batch Size used during training.
        :param max_history: Max number of context sentences.
        :param num_candidates: Number of distractors used during training.
        :param personality_permutations: Max number of personality permutations.
            (aka different persona sentence orders)
        """

        pretrained_model: str = "gpt2"
        # Optimizer
        learning_rate: float = 6.25e-5
        lm_coef: float = 2.0
        mc_coef: float = 1.0

        # Data configs
        dataset_path: str = ""
        # Training details
        batch_size: int = 2
        max_history: int = 2
        personality_permutations: int = 2
        num_candidates: int = 4

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams = hparams
        # GPT2 is going to be frozen and fixed!
        # because of that we hide it inside the DataModule
        self.gpt2 = GPT2DoubleHeadsModel.from_pretrained(self.hparams.pretrained_model)
        self.tokenizer = Tokenizer(self.hparams.pretrained_model)
        # Resize embeddings to include the added tokens
        self.gpt2.resize_token_embeddings(self.tokenizer.vocab_size)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.hparams.learning_rate, correct_bias=True
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def forward(
        self,
        input_ids: torch.Tensor,
        mc_token_ids: torch.Tensor,
        lm_labels: torch.Tensor = None,
        mc_labels: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
    ) -> GPT2DoubleHeadsModelOutput:
        return self.gpt2(
            input_ids,
            token_type_ids=token_type_ids,
            mc_token_ids=mc_token_ids,
            mc_labels=mc_labels,
            labels=lm_labels,
            return_dict=True,
        )

    def training_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Runs one training step. This usually consists in the forward function followed
            by the loss function.

        :param batch: The output of your dataloader.
        :param batch_nb: Integer displaying which batch this is
        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        # input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        output = self.forward(*batch)
        loss_val = (
            output.loss * self.hparams.lm_coef + output.mc_loss * self.hparams.mc_coef
        )

        # Language Modeling Negative Log-Likelihood
        nll = output.loss

        # Multiple-Choice Prediction.
        mc_pred, mc_target = torch.topk(output.mc_logits, 1)[1].view(-1), batch[3]
        acc = accuracy(mc_pred, mc_target)

        return {
            "loss": loss_val,
            "log": {"train_loss": loss_val, "train_nll": nll, "train_acc": acc},
        }

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Similar to the training step but with the model in eval mode.

        :returns: dictionary passed to the validation_end function.
        """
        # input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        output = self.forward(*batch)
        loss_val = (
            output.loss * self.hparams.lm_coef + output.mc_loss * self.hparams.mc_coef
        )

        # Language Modeling Negative Log-Likelihood
        nll = output.loss

        # Multiple-Choice Prediction.
        mc_pred, mc_target = torch.topk(output.mc_logits, 1)[1].view(-1), batch[3]
        acc = accuracy(mc_pred, mc_target)

        output = {"val_loss": loss_val, "val_nll": nll, "val_acc": acc}
        return output

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """
        # Average all metrics
        metrics = {
            "val_loss": torch.stack([x["val_loss"] for x in outputs]).mean(),
            "val_nll": torch.stack([x["val_nll"] for x in outputs]).mean(),
            "val_acc": torch.stack([x["val_acc"] for x in outputs]).mean(),
        }
        return {
            "progress_bar": metrics,
            "log": metrics,
        }

    @classmethod
    def from_experiment(cls, experiment_folder: str):
        """Function that loads the model from an experiment folder.

        :param experiment_folder: Path to the experiment folder.

        :return:Pretrained model.
        """
        hparams_file = experiment_folder + "hparams.yaml"
        hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)

        checkpoints = [
            file for file in os.listdir(experiment_folder) if file.endswith(".ckpt")
        ]
        checkpoint_path = experiment_folder + checkpoints[-1]
        model = cls.load_from_checkpoint(
            checkpoint_path, hparams=Namespace(**hparams), strict=True
        )
        # Make sure model is in prediction mode
        model.eval()
        model.freeze()
        return model

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
        return self.gpt2.generate(
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
