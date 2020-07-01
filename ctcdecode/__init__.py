import numpy as np
from ._ext import ctc_decode
from typing import List, NamedTuple, Optional, Tuple


Candidate = NamedTuple('Candidate', [('value', List[int]), ('log_prob', float)])


class CTCBeamDecoder:
    def __init__(
        self,
        cutoff_top_n = 40,
        cutoff_prob = 1.0,
        beam_width = 100,
        num_processes = 4,
        blank_id = 0,
        log_probs_input = False
    ):
        self.cutoff_top_n = cutoff_top_n
        self.beam_width = beam_width
        self.num_processes = num_processes
        self.blank_id = blank_id
        self.log_probs_input = log_probs_input
        self.cutoff_prob = cutoff_prob

    def decode(self, probs, seq_lens=None):
        """
        Input: probs, seq_lens numpy array
        """
        # We expect batch x seq x label_size
        batch_size, max_seq_len = probs.shape[:2]
        if seq_lens is None:
            seq_lens = np.full((batch_size, ), max_seq_len, dtype=np.int32)

        out = ctc_decode.beam_decode(
            probs,
            seq_lens,
            self.beam_width,
            self.num_processes,
            self.cutoff_prob,
            self.cutoff_top_n,
            self.blank_id,
            self.log_probs_input,
        )

        # convert to named tuples
        outt = []
        for batch_out in out:
            outt.append([Candidate(value, -score) for value, score in batch_out])
        return outt
