import torch
from ._ext import ctc_decode


class CTCBeamDecoder:
    def __init__(
        self,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=4,
        blank_id=0,
        log_probs_input=False
    ):
        self.cutoff_top_n = cutoff_top_n
        self.beam_width = beam_width
        self.num_processes = num_processes
        self.blank_id = blank_id
        self.log_probs_input = log_probs_input
        self.cutoff_prob = cutoff_prob

    def decode(self, probs, seq_lens=None):
        # We expect batch x seq x label_size
        probs = probs.cpu().float()
        batch_size, max_seq_len = probs.size(0), probs.size(1)
        if seq_lens is None:
            seq_lens = torch.IntTensor(batch_size).fill_(max_seq_len)
        else:
            seq_lens = seq_lens.cpu().int()

        return ctc_decode.beam_decode(
            probs, seq_lens, self.beam_width, self.num_processes, self.cutoff_prob,
            self.cutoff_top_n, self.blank_id, self.log_probs_input
        )
