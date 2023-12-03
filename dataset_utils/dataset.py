import os
from functools import partial
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from sentencepiece import SentencePieceProcessor

from jax import Array
from torch.utils.data import Dataset

def get_ext(path: Union[str, os.PathLike]) -> str:
    _, ext = os.path.splitext(path)
    return ext

class TextDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, os.PathLike],
        sp_processor: SentencePieceProcessor,
        col_name: Optional[str] = None,
    ):
        match get_ext(data_path):
            case ".csv":
                if not isinstance(col_name, str):
                    raise ValueError(f"`col_name` must be specified for `{data_path}`")
                df = pd.read_csv(data_path, header=None, names=[col_name], dtype=str)
                df.dropna(inplace=True)
                self.data = df[col_name].to_list()
            case ".txt":
                self.data = []
                with open(data_path, "r", encoding="utf-8") as f:
                    for text in f:
                        self.data.append(text)
            case _:
                raise ValueError("only csv or txt files are supported")

        self.encode = partial(sp_processor.encode, add_bos=True, add_eos=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> List[int]:
        text = self.data[index]
        return self.encode(text)

def make_collate_fn(
    max_seq_len: int,
    pad_id: int,
    batch_size: int,
) -> Callable[[List[List[int]]], Array]:
    def collate_fn(token_batch: List[List[int]]) -> Array:
        padded_batch = []
        for tokens in token_batch:
            num_tokens = len(tokens)
            num_pads = max(0, max_seq_len + 1 - num_tokens)
            padded_tokens = np.pad(
                np.array(tokens),
                (0, num_pads),
                mode="constant",
                constant_values=pad_id,
            )
            padded_batch.append(padded_tokens[:max_seq_len + 1])

        batch = np.vstack(padded_batch).astype(np.int32)
        num_batch_pads = batch_size - batch.shape[0]
        if num_batch_pads > 0:
            batch_pads = np.full((num_batch_pads, *batch.shape[1:]), pad_id, dtype=np.int32)
            batch = np.vstack((batch, batch_pads))
        return batch

    return collate_fn
