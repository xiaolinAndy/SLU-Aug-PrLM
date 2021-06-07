import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from transformers import (
    BartForMaskedLM,
    BartConfig,
    BartModel
)
from transformers.modeling_bart import _prepare_bart_decoder_inputs
from transformers.modeling_utils import PreTrainedModel, create_position_ids_from_input_ids, top_k_top_p_filtering, BeamHypotheses
sys.path.append("..")
from utils import crf

def _filter_out_falsey_values(tup):
    """Remove entries that are None or [] from an iterable."""
    return tuple(x for x in tup if isinstance(x, torch.Tensor) or x)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target, value_mask=None):
        mask = target.ne(-100)
        x = pred[mask]
        target = target[mask]
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)  # bs*classes
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)  # bs
        if torch.is_tensor(value_mask):
            smooth_prob = -logprobs * value_mask[mask].unsqueeze(-1)
            smooth_loss = smooth_prob.mean(dim=-1)
        else:
            smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class BartForGeneration(BartForMaskedLM):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.label_smoothing_layer = LabelSmoothingLoss(self.config.vocab_size, smoothing=0.1)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        lm_labels=None,
        value_mask=None
    ):
        outputs = self.model.forward(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
        )
        lm_logits = self.lm_head.forward(outputs[0])
        outputs = (lm_logits,) + outputs[1:]  # Add hidden states and attention if they are here
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            lm_logits = lm_logits.contiguous()
            #masked_lm_loss = loss_fct(lm_logits.reshape(-1, lm_logits.shape[-1]), lm_labels.reshape(-1))
            if torch.is_tensor(value_mask):
                masked_lm_loss = self.label_smoothing_layer(lm_logits.reshape(-1, self.config.vocab_size), lm_labels.reshape(-1), value_mask.reshape(-1))
            else:
                masked_lm_loss = self.label_smoothing_layer(lm_logits.reshape(-1, self.config.vocab_size),
                                                            lm_labels.reshape(-1))
            outputs = (masked_lm_loss,) + outputs
        # lm_loss, lm_logits, decoder_cached_states, dec_hidden, dec_attn
        return outputs

    def generate(
        self,
        input_ids=None,
        max_length=None,
        do_sample=True,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_ids=None,
        length_penalty=None,
        num_return_sequences=None,
    ):

        max_length = max_length if max_length is not None else self.config.max_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_ids = eos_token_ids if eos_token_ids is not None else self.config.eos_token_ids
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictely positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictely positive integer."
        assert temperature > 0, "`temperature` should be strictely positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_ids is None) or (
            isinstance(eos_token_ids, (list, tuple)) and ((isinstance(e, int) and e >= 0) for e in eos_token_ids)
        ), "`eos_token_ids` should be a positive integer or a list/tuple of positive integers."
        assert length_penalty > 0, "`length_penalty` should be strictely positive."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictely positive integer."

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # current position and vocab size
        cur_len = input_ids.shape[1]
        vocab_size = self.config.vocab_size

        if num_return_sequences != 1:
            # Expand input to num return sequences
            input_ids = input_ids.unsqueeze(1).expand(batch_size, num_return_sequences, cur_len)
            input_ids = input_ids.contiguous().view(
                batch_size * num_return_sequences, cur_len
            )  # (batch_size * num_return_sequences, cur_len)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
                length_penalty,
                num_beams,
                vocab_size,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                bos_token_id,
                eos_token_ids,
                effective_batch_size,
            )

        return output

    def _generate_no_beam_search(
        self,
        input_ids,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        bos_token_id,
        eos_token_ids,
        batch_size,
    ):
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        decoder_input_ids = input_ids.new(batch_size, 1).fill_(0)
        attn_values = input_ids.new(batch_size, sent_lengths.max().item(), input_ids.shape[1]).fill_(0).float()
        attn_enc_mask = input_ids.ne(pad_token_id)
        att_src_mask = attn_enc_mask & input_ids.ne(bos_token_id) & input_ids.ne(eos_token_ids[0])  # bs*src_len
        cur_len = 1

        while cur_len < max_length:
            #outputs = self(input_ids=input_ids, decoder_input_ids=decoder_input_ids, attention_mask=attn_enc_mask)
            outputs = self(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            next_token_logits = outputs[0][:, -1, :]
            #src, tgt = torch.topk(next_token_logits[1], 5)
            #print(src, tgt)
            attn = torch.mean(outputs[1][11][:, :, -1, :], dim=1)  # bs*src_len
            attn = attn.masked_fill(~att_src_mask, float("-inf"))
            #src_index = torch.argmax(attn, dim=-1) # bs
            attn_values[:, cur_len] = attn

            # if model has past, then set the past variable to speed up decoding
            if self._do_output_past(outputs):
                past = outputs[1]

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(decoder_input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_ids is not None:
                # pad finished sentences if eos_token_ids exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            decoder_input_ids = torch.cat([decoder_input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            if eos_token_ids is not None:
                for eos_token_id in eos_token_ids:
                    eos_in_sents = tokens_to_add == eos_token_id
                    # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                    is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                    sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1)
                    # unfinished_sents is set to zero if eos in sentence
                    unfinished_sents.mul_((~eos_in_sents).long())

            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # if there are different sentences lengths in the batch, some batches have to be padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
            # finished sents are filled with pad_token
            decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
        else:
            decoded = input_ids
        decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)


        for hypo_idx, hypo in enumerate(decoder_input_ids):
            decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

        return decoded, attn_values

# faster
class BartForGenerationTest(BartForMaskedLM):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.label_smoothing_layer = LabelSmoothingLoss(self.config.vocab_size, smoothing=0.1)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        lm_labels=None
    ):
        outputs = self.model.forward(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
        )
        lm_logits = self.lm_head.forward(outputs[0])
        outputs = (lm_logits,) + outputs[1:]  # Add hidden states and attention if they are here
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            lm_logits = lm_logits.contiguous()
            masked_lm_loss = loss_fct(lm_logits.reshape(-1, lm_logits.shape[-1]), lm_labels.reshape(-1))
            #masked_lm_loss = self.label_smoothing_layer(lm_logits.reshape(-1, self.config.vocab_size), lm_labels.reshape(-1))
            outputs = (masked_lm_loss,) + outputs
        # lm_loss, lm_logits, decoder_cached_states, dec_hidden, dec_attn
        return outputs

    def generate(
        self,
        input_ids=None,
        max_length=None,
        do_sample=True,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_ids=None,
        length_penalty=None,
        num_return_sequences=None,
    ):

        max_length = max_length if max_length is not None else self.config.max_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_ids = eos_token_ids if eos_token_ids is not None else self.config.eos_token_ids
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictely positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictely positive integer."
        assert temperature > 0, "`temperature` should be strictely positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_ids is None) or (
            isinstance(eos_token_ids, (list, tuple)) and ((isinstance(e, int) and e >= 0) for e in eos_token_ids)
        ), "`eos_token_ids` should be a positive integer or a list/tuple of positive integers."
        assert length_penalty > 0, "`length_penalty` should be strictely positive."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictely positive integer."

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # current position and vocab size
        cur_len = input_ids.shape[1]
        vocab_size = self.config.vocab_size

        if num_return_sequences != 1:
            # Expand input to num return sequences
            input_ids = input_ids.unsqueeze(1).expand(batch_size, num_return_sequences, cur_len)
            input_ids = input_ids.contiguous().view(
                batch_size * num_return_sequences, cur_len
            )  # (batch_size * num_return_sequences, cur_len)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
                length_penalty,
                num_beams,
                vocab_size,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                bos_token_id,
                eos_token_ids,
                effective_batch_size,
            )

        return output

    def _generate_no_beam_search(
        self,
        input_ids,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        bos_token_id,
        eos_token_ids,
        batch_size,
    ):
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        decoder_input_ids = input_ids.new(batch_size, 1).fill_(0)
        attn_values = input_ids.new(batch_size, sent_lengths.max().item(), input_ids.shape[1]).fill_(0).float()
        attn_enc_mask = input_ids.ne(pad_token_id)
        att_src_mask = attn_enc_mask & input_ids.ne(bos_token_id) & input_ids.ne(eos_token_ids[0])  # bs*src_len
        cur_len = 1

        # encoder cache:
        encoder_outputs = self.model.encoder.forward(input_ids=input_ids)#, attention_mask=attn_enc_mask)
        # toy test
        output_probs = []
        while cur_len < max_length:
            outputs = self(input_ids=input_ids, decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs)
            #outputs = self(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            next_token_logits = outputs[0][:, -1, :]
            src, tgt = torch.topk(F.softmax(next_token_logits[0], dim=-1), 20)
            output_probs.append([src, tgt])

            # if model has past, then set the past variable to speed up decoding
            if self._do_output_past(outputs):
                past = outputs[1]

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(decoder_input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_ids is not None:
                # pad finished sentences if eos_token_ids exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            decoder_input_ids = torch.cat([decoder_input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            if eos_token_ids is not None:
                for eos_token_id in eos_token_ids:
                    eos_in_sents = tokens_to_add == eos_token_id
                    # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                    is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                    sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1)
                    # unfinished_sents is set to zero if eos in sentence
                    unfinished_sents.mul_((~eos_in_sents).long())

            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # if there are different sentences lengths in the batch, some batches have to be padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
            # finished sents are filled with pad_token
            decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
        else:
            decoded = input_ids
        decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)


        for hypo_idx, hypo in enumerate(decoder_input_ids):
            decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

        return decoded, output_probs

    def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
        length_penalty,
        num_beams,
        vocab_size,
    ):
        """ Generate sequences for each example with beam search.
        """
        # Expand input to num beams
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, cur_len)
        input_ids = input_ids.contiguous().view(batch_size * num_beams, cur_len)  # (batch_size * num_beams, cur_len)
        decoder_input_ids = input_ids.new(batch_size, num_beams, 1).fill_(0)
        decoder_input_ids = decoder_input_ids.view(batch_size * num_beams, 1)
        cur_len = 1

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = None

        # done sentences
        done = [False for _ in range(batch_size)]

        # encoder cache:
        encoder_outputs = self.model.encoder.forward(input_ids=input_ids)  # , attention_mask=attn_enc_mask)

        while cur_len < max_length:
            outputs = self(input_ids=input_ids, decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs)
            scores = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            if self._do_output_past(outputs):
                past = outputs[1]

            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size * num_beams):
                    for previous_token in set(decoder_input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if scores[i, previous_token] < 0:
                            scores[i, previous_token] *= repetition_penalty
                        else:
                            scores[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                scores = top_k_top_p_filtering(
                    scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # Sample 2 next words for each beam (so we have some spare tokens and match output of greedy beam search)
                next_words = torch.multinomial(F.softmax(scores, dim=-1), num_samples=2)  # (batch_size * num_beams, 2)
                # Compute next scores
                _scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                _scores = torch.gather(_scores, -1, next_words)  # (batch_size * num_beams, 2)
                next_scores = _scores + beam_scores[:, None].expand_as(_scores)  # (batch_size * num_beams, 2)
                # Match shape of greedy beam search
                next_words = next_words.view(batch_size, 2 * num_beams)  # (batch_size, 2 * num_beams)
                next_scores = next_scores.view(batch_size, 2 * num_beams)  # (batch_size, 2 * num_beams)
            else:
                # do greedy beam search
                scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                assert scores.size() == (batch_size * num_beams, vocab_size)
                # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                _scores = _scores.view(batch_size, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
                next_scores, next_words = torch.topk(_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_words.size() == (batch_size, 2 * num_beams)

            # next batch beam content
            # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # if we are done with this sentence
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item()
                )
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                    assert (
                        eos_token_ids is not None and pad_token_id is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, score in zip(next_words[batch_idx], next_scores[batch_idx]):

                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size

                    # add to generated hypotheses if end of sentence or last iteration
                    if eos_token_ids is not None and word_id.item() in eos_token_ids:
                        generated_hyps[batch_idx].add(
                            decoder_input_ids[batch_idx * num_beams + beam_id, :cur_len].clone(), score.item()
                        )
                    else:
                        # add next predicted word if it is not eos_token
                        next_sent_beam.append((score, word_id, batch_idx * num_beams + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # update next beam content
                assert len(next_sent_beam) == num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = decoder_input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = decoder_input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch
            decoder_input_ids = decoder_input_ids[beam_idx, :]
            decoder_input_ids = torch.cat([decoder_input_ids, beam_words.unsqueeze(1)], dim=-1)

            # re-order internal states
            if past:
                reordered_past = []
                for layer_past in past:
                    # get the correct batch idx from layer past batch dim
                    # batch dim of `past` and `mems` is at 2nd position
                    reordered_layer_past = [layer_past[:, i].unsqueeze(1).clone().detach() for i in beam_idx]
                    reordered_layer_past = torch.cat(reordered_layer_past, dim=1)
                    # check that shape matches
                    assert reordered_layer_past.shape == layer_past.shape
                    reordered_past.append(reordered_layer_past)
                past = tuple(reordered_past)

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        for batch_idx in range(batch_size):
            # Add all open beam hypothesis to generated_hyps
            if not done[batch_idx]:
                for idx, score in zip(next_words[batch_idx], next_scores[batch_idx]):

                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size
                    generated_hyps[batch_idx].add(
                        decoder_input_ids[batch_idx * num_beams + beam_id, :cur_len].clone(), score.item()
                    )

        # select the best hypotheses
        sent_lengths = decoder_input_ids.new(batch_size)
        best = []
        all_beams = []
        sent_lengths = []
        for hypotheses in generated_hyps:
            for seq in hypotheses.beams:
                all_beams.append(seq[1])
                sent_lengths.append(len(seq[1]))
        sent_lengths = decoder_input_ids.new(sent_lengths)
        batch_size = batch_size * num_beams

        '''for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.beams, key=lambda x: x[0])[1]
            sent_lengths[i] = len(best_hyp)
            best.append(best_hyp)'''

        # shorter batches are filled with pad_token
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined"
            sent_max_len = min(sent_lengths.max().item() + 1, max_length)
            decoded = decoder_input_ids.new(batch_size, sent_max_len).fill_(pad_token_id)

            # fill with hypothesis and eos_token_id if necessary
            # change best with all_beams for test
            for i, hypo in enumerate(all_beams):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_length:
                    decoded[i, sent_lengths[i]] = eos_token_ids[0]
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_length for hypo in best)
            decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

        return decoded, None

class BartForSlotFiling(BartForMaskedLM):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.sf_layer = nn.Linear(self.model.shared.weight.shape[1], config.slot_size, bias=False)
        self.sf_decoder = copy.deepcopy(self.model.decoder)
        self.label_smoothing_layer = LabelSmoothingLoss(self.config.vocab_size, smoothing=0.1)

    def forward(
        self,
        input_ids,
        decoder_input_ids,
        lm_labels=None,
        slot_labels=None,
        attention_mask=None,
        decoder_attention_mask=None,
        decoder_cached_states=None
    ):
        # make masks if user doesn't supply
        decoder_input_ids, decoder_attn_mask = _prepare_bart_decoder_inputs(
            self.config, input_ids, decoder_input_ids=decoder_input_ids, decoder_attn_mask=decoder_attention_mask,
        )

        encoder_outputs = self.model.encoder.forward(input_ids=input_ids, attention_mask=attention_mask)
        lm_decoder_outputs = self.model.decoder.forward(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_attn_mask,
            decoder_cached_states=decoder_cached_states,
        )
        sf_decoder_outputs = self.sf_decoder.forward(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_attn_mask,
            decoder_cached_states=decoder_cached_states,
        )
        lm_decoder_outputs = _filter_out_falsey_values(lm_decoder_outputs)
        sf_decoder_outputs = _filter_out_falsey_values(sf_decoder_outputs)
        lm_logits = self.lm_head.forward(lm_decoder_outputs[0])
        slot_logits = self.sf_layer.forward(sf_decoder_outputs[0])

        outputs = (lm_logits, slot_logits,) + lm_decoder_outputs[1:]

        if torch.is_tensor(lm_labels):
            loss_fct = nn.CrossEntropyLoss()
            lm_logits = lm_logits.contiguous()
            #masked_lm_loss = loss_fct(lm_logits.reshape(-1, self.config.vocab_size), lm_labels.reshape(-1))
            masked_lm_loss = self.label_smoothing_layer(lm_logits.reshape(-1, self.config.vocab_size), lm_labels.reshape(-1))

            slot_logits = slot_logits.contiguous()
            slot_loss = loss_fct(slot_logits.reshape(-1, slot_logits.shape[-1]), slot_labels.reshape(-1))
            outputs = (masked_lm_loss, slot_loss,) + outputs

        return outputs

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        # current position / max lengths / length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        decoder_input_ids = input_ids.new(batch_size, 1).fill_(0)
        decoded_label = input_ids.new(batch_size, sent_lengths.max().item()).fill_(0)
        cur_len = 1

        while cur_len < max_length:
            #model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
            outputs = self(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            next_token_logits = outputs[0][:, -1, :]
            #value, next_token_id = torch.topk(F.softmax(next_token_logits), 5, dim=-1)
            #print(value[0], next_token_id[0], value[1], next_token_id[1])
            #return input_ids[0], next_token_id[0], input_ids[1], next_token_id[1]
            next_token_slot_logits = outputs[1][:, -1, :]
            next_token_slot_label = torch.argmax(next_token_slot_logits, dim=-1)
            decoded_label[:,cur_len] = next_token_slot_label

            # if model has past, then set the past variable to speed up decoding
            if self._do_output_past(outputs):
                past = outputs[1]

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(decoder_input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_ids is not None:
                # pad finished sentences if eos_token_ids exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            decoder_input_ids = torch.cat([decoder_input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            if eos_token_ids is not None:
                for eos_token_id in eos_token_ids:
                    eos_in_sents = tokens_to_add == eos_token_id
                    # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                    is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                    sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1)
                    # unfinished_sents is set to zero if eos in sentence
                    unfinished_sents.mul_((~eos_in_sents).long())

            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # if there are different sentences lengths in the batch, some batches have to be padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
            # finished sents are filled with pad_token
            decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
        else:
            decoded = input_ids
        decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)


        for hypo_idx, hypo in enumerate(decoder_input_ids):
            decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

        # slot prediction
        outputs = self(input_ids=input_ids, decoder_input_ids=decoded)
        slot_logits = outputs[1]
        next_token_slot_label = torch.argmax(slot_logits, dim=-1)
        decoded_label = next_token_slot_label

        return decoded, decoded_label

class BartForSlotFilingCRF(BartForMaskedLM):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.sf_layer = nn.Linear(self.model.shared.weight.shape[1], config.slot_size+2, bias=False)
        self.sf_decoder = copy.deepcopy(self.model.decoder)
        self.crf_layer = crf.CRF(config.slot_size)

    # new decoder for slot filling
    #def init_weight(self):
        #self.sf_decoder = copy.deepcopy(self.model.decoder)

    def forward(
        self,
        input_ids,
        decoder_input_ids,
        lm_labels=None,
        slot_labels=None,
        attention_mask=None,
        decoder_attention_mask=None,
        decoder_cached_states=None
    ):
        # make masks if user doesn't supply
        decoder_input_ids, decoder_attn_mask = _prepare_bart_decoder_inputs(
            self.config, input_ids, decoder_input_ids=decoder_input_ids, decoder_attn_mask=decoder_attention_mask,
        )

        encoder_outputs = self.model.encoder.forward(input_ids=input_ids, attention_mask=attention_mask)
        lm_decoder_outputs = self.model.decoder.forward(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_attn_mask,
            decoder_cached_states=decoder_cached_states,
        )
        sf_decoder_outputs = self.sf_decoder.forward(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_attn_mask,
            decoder_cached_states=decoder_cached_states,
        )
        lm_decoder_outputs = _filter_out_falsey_values(lm_decoder_outputs)
        sf_decoder_outputs = _filter_out_falsey_values(sf_decoder_outputs)
        lm_logits = self.lm_head.forward(lm_decoder_outputs[0])
        slot_logits = self.sf_layer.forward(sf_decoder_outputs[0])
        slot_mask = torch.ne(decoder_input_ids, 1).bool()
        total_len = torch.sum(slot_mask)

        outputs = (lm_logits, slot_logits,) + lm_decoder_outputs[1:]

        if torch.is_tensor(lm_labels):
            loss_fct = nn.CrossEntropyLoss()
            lm_logits = lm_logits.contiguous()
            masked_lm_loss = loss_fct(lm_logits.reshape(-1, self.config.vocab_size), lm_labels.reshape(-1))

            slot_logits = slot_logits.contiguous()
            slot_loss = self.crf_layer.neg_log_likelihood_loss(slot_logits, slot_mask, slot_labels)
            slot_loss /= total_len
            outputs = (masked_lm_loss, slot_loss,) + outputs

        return outputs

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        # current position / max lengths / length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        decoder_input_ids = input_ids.new(batch_size, 1).fill_(0)
        decoded_label = input_ids.new(batch_size, sent_lengths.max().item()).fill_(0)
        cur_len = 1

        while cur_len < max_length:
            #model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
            outputs = self(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            next_token_logits = outputs[0][:, -1, :]
            #value, next_token_id = torch.topk(F.softmax(next_token_logits), 5, dim=-1)
            #print(value[0], next_token_id[0], value[1], next_token_id[1])
            #return input_ids[0], next_token_id[0], input_ids[1], next_token_id[1]
            next_token_slot_logits = outputs[1][:, -1, :]
            next_token_slot_label = torch.argmax(next_token_slot_logits, dim=-1)
            decoded_label[:,cur_len] = next_token_slot_label

            # if model has past, then set the past variable to speed up decoding
            if self._do_output_past(outputs):
                past = outputs[1]

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(decoder_input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_ids is not None:
                # pad finished sentences if eos_token_ids exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            decoder_input_ids = torch.cat([decoder_input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            if eos_token_ids is not None:
                for eos_token_id in eos_token_ids:
                    eos_in_sents = tokens_to_add == eos_token_id
                    # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                    is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                    sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1)
                    # unfinished_sents is set to zero if eos in sentence
                    unfinished_sents.mul_((~eos_in_sents).long())

            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # if there are different sentences lengths in the batch, some batches have to be padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
            # finished sents are filled with pad_token
            decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
        else:
            decoded = input_ids
        decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)


        for hypo_idx, hypo in enumerate(decoder_input_ids):
            decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

        # crf prediction
        outputs = self(input_ids=input_ids, decoder_input_ids=decoded)
        slot_logits = outputs[1]
        slot_mask = torch.ne(decoded, 1).bool()
        path_score, best_path = self.crf_layer._viterbi_decode(slot_logits, slot_mask)
        #print(best_path.shape, decoded_label.shape)
        decoded_label = best_path

        return decoded, decoded_label