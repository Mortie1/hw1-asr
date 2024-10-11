import re
from collections import defaultdict
from string import ascii_lowercase
from typing import List

import torch
from numpy.typing import NDArray
from torchaudio.models.decoder import (
    ctc_decoder,
    cuda_ctc_decoder,
    download_pretrained_files,
)

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier

# i know it's ugly but i spent ~6 hours debugging installation of ctcdecoder and yet not succeeded
# so let hardcode begin

bsdecoder = ctc_decoder(
    lexicon=None,
    tokens=[""] + list(ascii_lowercase + " ") + ["|", "<unk>"],
    beam_size=50,
    # lm=download_pretrained_files("librispeech-3-gram").lm,
    # lm="3-gram.pruned.3e-7.arpa",
    blank_token="",
)

cuda_decoder = cuda_ctc_decoder(
    tokens=[""] + list(ascii_lowercase + " "),
    beam_size=50,
    blank_skip_threshold=0.99,
)


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, beam_size: int = 50, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        # self.bsdecoder = ctc_decoder(
        #     lexicon=None,
        #     tokens=self.vocab + ["|", "<unk>"],
        #     beam_size=beam_size,
        #     blank_token=self.EMPTY_TOK,
        # )

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        """
        Decodes CTC output

        Args:
            inds (list): list of tokens
        Returns:
            text (str): decoded text
        """
        decoded_text = []
        for ind in inds:
            if ind != self.char2ind[self.EMPTY_TOK]:
                if len(decoded_text) == 0 or decoded_text[-1] != ind:
                    decoded_text.append(ind)
        return "".join([self.ind2char[int(ind)] for ind in decoded_text])

    def ctc_beamsearch(
        self,
        log_probs: torch.Tensor,
        log_probs_lengths: torch.Tensor,
        beam_size: int = 50,
    ) -> List[str]:
        """
        Decodes model output via beamsearch

        Args:
            log_probs (NDArray): 2d tensor of characters log_probs. shape: (batch_size, time, vocab_size)
            log_probs_lengths (NDArray): 1d tensor of length of each sequence in batch. shape: (batch_size,)
            beam_size (int): width of searching beam
        Returns:
            text (str): decoded text
        """
        # LEGACY HANDMADE CODE
        # # code from seminar
        # def truncate_paths(dp, beam_size):
        #     return dict(
        #         sorted(list(dp.items()), key=lambda x: x[1], reverse=True)[:beam_size]
        #     )

        # def expand_and_merge_paths(dp, next_token_probs):
        #     new_dp = defaultdict(float)
        #     for ind, next_token_prob in enumerate(next_token_probs):
        #         cur_char = self.ind2char[ind]
        #         for (prefix, last_char), v in dp.items():
        #             if last_char == cur_char:
        #                 new_prefix = prefix
        #             else:
        #                 if cur_char != self.EMPTY_TOK:
        #                     new_prefix = prefix + cur_char
        #                 else:
        #                     new_prefix = prefix
        #             new_dp[(new_prefix, cur_char)] += v * next_token_prob
        #     return new_dp

        # dp = {("", self.EMPTY_TOK): 1.0}
        # for prob in probs:
        #     dp = expand_and_merge_paths(dp, prob)
        #     dp = truncate_paths(dp, beam_size)
        # result = [
        #     (prefix, proba)
        #     for (prefix, _), proba in sorted(
        #         dp.items(), key=lambda x: x[1], reverse=True
        #     )
        # ][0][0]
        # lm_preds = bsdecoder(
        #     torch.nn.functional.pad(log_probs.cpu(), (0, 2), "constant", -1e9),
        #     log_probs_lengths,
        # )
        lm_preds = cuda_decoder(log_probs, log_probs_lengths)
        lm_preds = [
            "".join(
                bsdecoder.idxs_to_tokens(
                    torch.as_tensor(
                        max(preds_list, key=lambda x: x.score).tokens
                    ).long()
                )
            )
            for preds_list in lm_preds
        ]
        return lm_preds

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
