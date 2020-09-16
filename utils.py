# -*- coding: utf-8 -*-
from argparse import Namespace

import torch


class Config:
    def __init__(self, initial_data: dict) -> None:
        for key in initial_data:
            if hasattr(self, key):
                setattr(self, key, initial_data[key])

    def namespace(self) -> Namespace:
        return Namespace(
            **{
                name: getattr(self, name)
                for name in dir(self)
                if not callable(getattr(self, name)) and not name.startswith("__")
            }
        )


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)
