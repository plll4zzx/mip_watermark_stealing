# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import collections
from math import sqrt

import scipy.stats

import torch
from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor

from nltk.util import ngrams

from normalizers import normalization_strategy_lookup
from GensimModel import GensimModel


class WatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",  # mostly unused/always default
        hash_key: int = 15485863,  # just a large prime number to create a rng seed with sufficient bit width
        select_green_tokens: bool = True,
        wm_level='token',
        finit_key_num=10,
        model_key=123,
        save_green=False,
        tokenizer=None,
        simi_num=8000,simi_num_for_token=10,
        model_name='',
    ):

        # watermarking parameters
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key
        self.select_green_tokens = select_green_tokens

        self.wm_level=wm_level
        self.finit_key_num=finit_key_num
        self.green_list_dict={}
        self.model_key=model_key
        self.model_name=model_name

        self.key_list=[]
        
        if self.wm_level=='model_simi':
            self.simi_num=int(self.vocab_size*self.gamma/2*3)
            self.simi_num_for_token=simi_num_for_token
            self.gensimi=GensimModel()
            self.tokenizer=tokenizer

    def __del__(self):
        green_list_dict={
            key:self.green_list_dict[key].cpu().tolist()
            for key in self.green_list_dict
        }
        from get_greenlist import get_greenlist_filename
        from utli import save_json
        import os
        filename=get_greenlist_filename(self.key_list, self.gamma, self.model_name+'_'+self.wm_level)
        file_path=os.path.join('saved_data', filename)
        save_json(data=green_list_dict, file_path=file_path)

    def _seed_rng(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None:
        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme

        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            if 'token' == self.wm_level:
                prev_token = input_ids[-1].item()
            elif 'sentence' == self.wm_level:
                try:
                    prev_token = input_ids[20].item()
                except:
                    prev_token = input_ids[1].item()
            elif 'sentence_fi' == self.wm_level:
                try:
                    prev_token = input_ids[20].item()
                except:
                    prev_token = input_ids[1].item()
                prev_token = (prev_token%self.finit_key_num)+1
            else:
                prev_token = self.model_key
            # prev_token = input_ids[-1].item()
            self.rng.manual_seed(self.hash_key * prev_token)
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        if prev_token not in self.key_list:
            self.key_list.append(prev_token)
        return prev_token

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        # seed the rng using the previous tokens/prefix
        # according to the seeding_scheme
        key=self._seed_rng(input_ids)

        if key not in self.green_list_dict:
            greenlist_size = int(self.vocab_size * self.gamma)
            vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)
            if self.select_green_tokens:  # directly
                greenlist_ids = vocab_permutation[:greenlist_size]  # new
            else:  # select green via red
                greenlist_ids = vocab_permutation[(self.vocab_size - greenlist_size) :]  # legacy behavior
            self.green_list_dict[key]=greenlist_ids
        return self.green_list_dict[key]
    
    def _get_simi_greenlist_ids(self, input_ids: torch.LongTensor, device='cuda'):
        
        key=self._seed_rng(input_ids)

        if key not in self.green_list_dict:
            greenlist_size = int(self.vocab_size * self.gamma)
            vocab_permutation = torch.randperm(self.vocab_size, device=device, generator=self.rng)#
            greenlist_ids_list=[]
            for idx in range(len(vocab_permutation.tolist())):
                if len(greenlist_ids_list)>=greenlist_size or idx>=self.simi_num:
                    break
                token_idx = vocab_permutation[idx].item()
                token=self.tokenizer.decode(token_idx)
                if len(token)<=3:
                    continue
                if token in ['that', 'the', 'to', 'of', 'is', 'are','be','on','in','it','an','and','for']:
                    continue
                simi_word_list=self.gensimi.find_simi_words(token, simi_num=self.simi_num_for_token)+[token]
                simi_index_list=[
                    token_id
                    for word in simi_word_list
                    for token in self.tokenizer.tokenize(word)
                    for token_id in self.tokenizer.encode(token)
                    if token_id not in greenlist_ids_list
                    if len(self.tokenizer.decode(token_id))>3
                ]
                simi_index_list=list(set(simi_index_list))
                greenlist_ids_list=greenlist_ids_list+simi_index_list
            greenlist_ids_list=list(set(greenlist_ids_list))
            for idx in range(len(vocab_permutation.tolist())-1,0,-1):
                if len(greenlist_ids_list)>greenlist_size:
                    break
                tmp_token=vocab_permutation[idx].item()
                if tmp_token in greenlist_ids_list:
                    continue
                greenlist_ids_list.append(tmp_token)
            if len(greenlist_ids_list)>greenlist_size:
                greenlist_ids=torch.tensor(greenlist_ids_list[0:greenlist_size]).to(device)
            else:
                greenlist_ids=torch.tensor(greenlist_ids_list).to(device)
            greenlist_ids=torch.cat(
                (
                    greenlist_ids,
                    vocab_permutation[(self.vocab_size - (greenlist_size-len(greenlist_ids_list))) :]
                )
            )
            greenlist_ids=greenlist_ids.to(torch.int64)
            self.green_list_dict[key]=greenlist_ids
        return self.green_list_dict[key]


class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    def __init__(self, *args, **kwargs):
        # self.wm_level=wm_level

        super().__init__(*args, **kwargs)

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        # TODO lets see if we can lose this loop
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # this is lazy to allow us to colocate on the watermarked model's device
        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)

        # NOTE, it would be nice to get rid of this batch loop, but currently,
        # the seed and partition operations are not tensor/vectorized, thus
        # each sequence in the batch needs to be treated separately.
        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            if self.wm_level=='model_simi':
                greenlist_ids = self._get_simi_greenlist_ids(input_ids[b_idx])
            else:
                greenlist_ids = self._get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)
        return scores


class WatermarkDetector(WatermarkBase):
    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"]
        ignore_repeated_bigrams: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        if self.seeding_scheme == "simple_1":
            self.min_prefix_len = 1
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {self.seeding_scheme}")

        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(normalization_strategy))

        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        if self.ignore_repeated_bigrams:
            assert self.seeding_scheme == "simple_1", "No repeated bigram credit variant assumes the single token seeding scheme."

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value

    def _score_sequence(
        self,
        input_ids: Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_p_value: bool = True,
    ):
        if self.ignore_repeated_bigrams:
            # Method that only counts a green/red hit once per unique bigram.
            # New num total tokens scored (T) becomes the number unique bigrams.
            # We iterate over all unqiue token bigrams in the input, computing the greenlist
            # induced by the first token in each, and then checking whether the second
            # token falls in that greenlist.
            assert return_green_token_mask is False, "Can't return the green/red mask when ignoring repeats."
            bigram_table = {}
            token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
            freq = collections.Counter(token_bigram_generator)
            num_tokens_scored = len(freq.keys())
            for idx, bigram in enumerate(freq.keys()):
                prefix = torch.tensor([bigram[0]], device=self.device)  # expects a 1-d prefix tensor on the randperm device
                greenlist_ids = self._get_greenlist_ids(prefix)
                bigram_table[bigram] = True if bigram[1] in greenlist_ids else False
            green_token_count = sum(bigram_table.values())
        else:
            num_tokens_scored = len(input_ids) - self.min_prefix_len
            if num_tokens_scored < 1:
                raise ValueError(
                    (
                        f"Must have at least {1} token to score after "
                        f"the first min_prefix_len={self.min_prefix_len} tokens required by the seeding scheme."
                    )
                )
            # Standard method.
            # Since we generally need at least 1 token (for the simplest scheme)
            # we start the iteration over the token sequence with a minimum
            # num tokens as the first prefix for the seeding scheme,
            # and at each step, compute the greenlist induced by the
            # current prefix and check if the current token falls in the greenlist.
            green_token_count, green_token_mask = 0, []
            for idx in range(self.min_prefix_len, len(input_ids)):
                curr_token = input_ids[idx]
                greenlist_ids = self._get_greenlist_ids(input_ids[:idx])
                if curr_token in greenlist_ids:
                    green_token_count += 1
                    green_token_mask.append(True)
                else:
                    green_token_mask.append(False)

        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        if return_z_score:
            score_dict.update(dict(z_score=self._compute_z_score(green_token_count, num_tokens_scored)))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask))

        return score_dict

    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        **kwargs,
    ) -> dict:

        assert (text is not None) ^ (tokenized_text is not None), "Must pass either the raw or tokenized string"
        if return_prediction:
            kwargs["return_p_value"] = True  # to return the "confidence":=1-p of positive detections

        # run optional normalizers on text
        for normalizer in self.normalizers:
            text = normalizer(text)
        if len(self.normalizers) > 0:
            print(f"Text after normalization:\n\n{text}\n")

        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            # try to remove the bos_tok at beginning if it's there
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]

        # call score method
        output_dict = {}
        score_dict = self._score_sequence(tokenized_text, **kwargs)
        if return_scores:
            output_dict.update(score_dict)
        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert z_threshold is not None, "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        return output_dict
