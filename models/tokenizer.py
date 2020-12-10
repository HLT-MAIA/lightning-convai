# -*- coding: utf-8 -*-
r""" 
Text Tokenizer
==============
    Wrapper around Hugging Face tokenizer.
"""
import torch
from torchnlp.encoders.text.text_encoder import TextEncoder
from transformers import AutoTokenizer

SPECIAL_TOKENS = ["<bos>", "<eos>", "<user>", "<assistant>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<user>", "<assistant>"],
}


class Tokenizer(TextEncoder):
    """Wrapper around Hugging Face tokenizer.

    :param pretrained_model: Hugging Face pretrained model.
    """

    def __init__(self, pretrained_model) -> None:
        self.enforce_reversible = False
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        orig_vocab = self.tokenizer.vocab_size
        num_added_tokens = self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
        self.vocab_size = orig_vocab + num_added_tokens

        self.pad_index = self.tokenizer.pad_token_id
        self.eos_index = self.tokenizer.eos_token_id
        self.bos_index = self.tokenizer.bos_token_id
        self.vocab = self.tokenizer.get_vocab()
        self.user_index = self.vocab["<user>"]
        self.assistant_index = self.vocab["<assistant>"]

    def encode(self, sequence: str) -> torch.Tensor:
        """Encodes a 'sequence'.
        :param sequence: String 'sequence' to encode.

        :return: torch.Tensor with Encoding of the `sequence`.
        """
        sequence = TextEncoder.encode(self, sequence)
        return self.tokenizer(sequence, add_special_tokens=False)["input_ids"]

    def decode(
        self,
        tensor: torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        return self.tokenizer.decode(
            tensor, skip_special_tokens, clean_up_tokenization_spaces
        )
