# -*- coding: utf-8 -*-
r""" 
Base Assistant
==================
    Base model to create a dialog Assistant.
"""
import os
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from transformers import AdamW
from transformers.file_utils import ModelOutput
from dataclasses import dataclass

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from utils import Config


@dataclass
class AssitantModelOutput(ModelOutput):
    lm_loss: Optional[torch.FloatTensor] = None
    mc_loss: Optional[torch.FloatTensor] = None
    lm_logits: torch.FloatTensor = None
    mc_logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class AssistantBase(pl.LightningModule):
    """T5 Model implementing the PyTorch Lightning interface that can be used to
        train a taskmaster Chatbot.

    :param hparams: ArgumentParser containing the hyperparameters.
    """

    class ModelConfig(Config):
        """The ModelConfig class is used to define Model settings.

        :param pretrained_model: Pretrained GPT2 model to be used.
        :param learning_rate: Learning Rate used during training.
        :param lm_coef: Weight assigned to the LM loss.
        :param mc_coef: Weight assigned to the Multiple-Choice loss.
        :param train_data: Path to a json file containing the train data.
        :param valid_data: Path to a json file containing the validation data.
        :param batch_size: Batch Size used during training.
        :param max_history: Max number of context sentences.
        :param num_candidates: Number of distractors used during training.
        """

        model: str = "AssistantT5"
        pretrained_model: str = "t5-small"

        # Optimizer
        learning_rate: float = 6.25e-5
        lm_coef: float = 1.0
        mc_coef: float = 0.5

        # Data configs
        train_data: str = "data/taskmaster/taskmaster-train.json"
        valid_data: str = "data/taskmaster/taskmaster-valid.json"

        # Training details
        batch_size: int = 2
        max_history: int = 2
        num_candidates: int = 4

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.hparams.learning_rate, correct_bias=True
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def forward(self, *args, **kwargs) -> AssitantModelOutput:
        pass

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
        output = self.forward(**batch)
        loss_val = (
            output.lm_loss * self.hparams.lm_coef
            + output.mc_loss * self.hparams.mc_coef
        )
        # Multiple-Choice Prediction.
        mc_pred, mc_target = (
            torch.topk(output.mc_logits, 1)[1].view(-1),
            batch["mc_labels"],
        )
        acc = accuracy(mc_pred, mc_target)
        return {
            "loss": loss_val,
            "log": {
                "total_loss": loss_val,
                "mc_loss": output.mc_loss,
                "lm_loss": output.lm_loss,
                "acc": acc,
            },
        }

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Similar to the training step but with the model in eval mode.

        :returns: dictionary passed to the validation_end function.
        """
        output = self.forward(**batch)
        loss_val = (
            output.lm_loss * self.hparams.lm_coef
            + output.mc_loss * self.hparams.mc_coef
        )
        # Multiple-Choice Prediction.
        mc_pred, mc_target = (
            torch.topk(output.mc_logits, 1)[1].view(-1),
            batch["mc_labels"],
        )
        acc = accuracy(mc_pred, mc_target)
        return {
            "val_total_loss": loss_val,
            "val_mc_loss": output.mc_loss,
            "val_lm_loss": output.lm_loss,
            "val_acc": acc,
        }

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
            "val_total_loss": torch.stack(
                [x["val_total_loss"] for x in outputs]
            ).mean(),
            "val_mc_loss": torch.stack([x["val_mc_loss"] for x in outputs]).mean(),
            "val_lm_loss": torch.stack([x["val_lm_loss"] for x in outputs]).mean(),
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
        hparams_file = os.path.join(experiment_folder, "hparams.yaml")
        hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)

        checkpoints = [
            file for file in os.listdir(experiment_folder) if file.endswith(".ckpt")
        ]
        checkpoint_path = os.path.join(experiment_folder, checkpoints[-1])
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
        pass
