import torch
from ._ext import ctc_decode
from typing import List, NamedTuple, Optional, Tuple


Candidate = NamedTuple('Candidate', [('value', List[int]), ('score', float)])


class CTCBeamDecoder:
    def __init__(
        self,
        cutoff_top_n: int = 40,
        cutoff_prob: float = 1.0,
        beam_width: int = 100,
        num_processes: int = 4,
        blank_id: int = 0,
        log_probs_input: bool = False
    ):
        self.cutoff_top_n = cutoff_top_n
        self.beam_width = beam_width
        self.num_processes = num_processes
        self.blank_id = blank_id
        self.log_probs_input = log_probs_input
        self.cutoff_prob = cutoff_prob

    def decode(
        self,
        probs: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> List[List[Candidate]]:

        # We expect batch x seq x label_size
        probs = probs.cpu().float()
        batch_size, max_seq_len = probs.shape[:2]
        if seq_lens is None:
            seq_lens = torch.full((batch_size, ), max_seq_len, dtype=torch.int)
        else:
            seq_lens = seq_lens.cpu().int()

        out: List[List[Tuple[List[int], float]]] = ctc_decode.beam_decode(
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
            outt.append([Candidate(*c) for c in batch_out])
        return outt
