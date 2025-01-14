from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from ._ext import ctc_decode


class Candidate(NamedTuple):
    value: list[int]
    log_prob: float


class CTCBeamDecoder:
    def __init__(
        self,
        cutoff_top_n: int = 40,
        cutoff_prob: float = 1.0,
        beam_width: int = 100,
        num_processes: int = 4,
        blank_id: int = 0,
    ):
        self.cutoff_top_n = cutoff_top_n
        self.beam_width = beam_width
        self.num_processes = num_processes
        self.blank_id = blank_id
        self.cutoff_prob = cutoff_prob

    def decode(
        self, log_probs: NDArray[np.float32], seq_lens: NDArray[np.integer] | None = None
    ) -> list[list[Candidate]]:
        """
        Input: probs, seq_lens numpy array
        """
        # We expect batch x seq x label_size
        batch_size, max_seq_len = log_probs.shape[:2]
        if seq_lens is None:
            seq_lens = np.full((batch_size,), max_seq_len, dtype=np.int32)

        out = ctc_decode.beam_decode(
            log_probs,
            seq_lens,
            self.beam_width,
            self.num_processes,
            self.cutoff_prob,
            self.cutoff_top_n,
            self.blank_id,
        )

        # convert to named tuples
        return [[Candidate(value, -score) for value, score in batch_out] for batch_out in out]
