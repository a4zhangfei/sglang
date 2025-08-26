from __future__ import annotations

from typing import TYPE_CHECKING, List, Type, Callable

import numpy as np
import torch
import triton
import threading

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

if TYPE_CHECKING:
    from python.sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch

from sglang.srt.speculative.eagle_utils import (
    assign_req_to_token_pool,
    assign_req_to_token_pool_async,
    create_flashinfer_kv_indices_triton,
)
from sgl_kernel import lookahead_verify_tree_greedy

@dataclass
class LookaheadVerifyInput:
    def __init__(
        self,
        draft_token: torch.Tensor,
        tree_mask: torch.Tensor,
        positions: torch.Tensor,
        retrive_index: torch.Tensor,
        retrive_next_token: torch.Tensor,
        retrive_next_sibling: torch.Tensor,

        accept_length: torch.Tensor,
        accept_token_ids: torch.Tensor,
        last_verified_ids: torch.Tensor,
        flatten_index: torch.Tensor,
        total_accept_num: torch.Tensor,

        draft_token_num: torch.Tensor,
        draft_token_num_cpu: int,
    ):
        self.draft_token = draft_token
        self.custom_mask = tree_mask
        self.positions = positions
        self.retrive_index = retrive_index
        self.retrive_next_token = retrive_next_token
        self.retrive_next_sibling = retrive_next_sibling

        self.accept_length = accept_length
        self.accept_token_ids = accept_token_ids
        self.last_verified_ids = last_verified_ids
        self.flatten_index = flatten_index
        self.total_accept_num = total_accept_num

        self.draft_token_num = draft_token_num
        self.draft_token_num_cpu = draft_token_num_cpu
        self.device = draft_token_num.device
        self.process_done: threading.Event|None = None

        # result
        self.accept_token_ids_cpu: torch.Tensor|None = None

    def prepare_for_verify(self, batch: ScheduleBatch):
        batch.input_ids = self.draft_token
        batch.out_cache_loc = batch.alloc_token_slots(batch.input_ids.numel())
        bs = batch.seq_lens.numel()
        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + self.draft_token_num,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        batch_size = len(req_pool_indices)

        cum_kv_seq_len = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=self.device
        )

        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        self.qo_indptr = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=self.device
        )
        self.qo_indptr[1:] = torch.cumsum(self.draft_token_num, dim=0)

        kv_indices = torch.empty(cum_kv_seq_len[-1], dtype=torch.int32, device=self.device)

        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices, cum_kv_seq_len, self.qo_indptr, self.custom_mask

    def post_process(self, batch: ScheduleBatch, logits_output: torch.Tensor):
        accept_index_flatten = self.flatten_index[:self.total_accept_num]
        evict_index_flatten = self.flatten_index[self.total_accept_num:]

        mem_need_free_idx = batch.out_cache_loc[evict_index_flatten]
        batch.token_to_kv_pool_allocator.free(mem_need_free_idx)

        logits_output.next_token_logits = logits_output.next_token_logits[
            accept_index_flatten
        ]

        accept_token_ids_cpu = self.accept_token_ids_cpu.tolist()
        for i, (req, accept_token_ids) in enumerate(zip(batch.reqs, accept_token_ids_cpu)):
            for accept_token_id in accept_token_ids:
                if accept_token_id < 0:
                    break
                req.output_ids.append(accept_token_id)
                batch.seq_lens_sum += 1
            req.check_finished()


    def verify(self, batch: ModelWorkerBatch, logits_output: torch.Tensor) -> torch.Tensor:
        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).to(torch.int32)
        lookahead_verify_tree_greedy(
            accept_token_num=self.accept_length,  # mutable
            accept_token_ids=self.accept_token_ids, # mutable
            last_verified_ids=self.last_verified_ids, # mutable
            flatten_index=self.flatten_index, # mutable
            total_accept_num=self.total_accept_num, # mutable
            candidates=self.draft_token,
            retrive_index=self.retrive_index,
            retrive_next_token=self.retrive_next_token,
            retrive_next_sibling=self.retrive_next_sibling,
            target_predict=target_predict,
            eos_token_id=batch.eos_id,
        )

        bs = self.retrive_index.shape[0]
        assign_req_to_token_pool_async[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + self.accept_length,
            batch.out_cache_loc,
            self.flatten_index,
            batch.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )

        batch.seq_lens.add_(self.accept_length)
        self.accept_token_ids_cpu = self.accept_token_ids.to("cpu", non_blocking=True)

        return logits_output, self.last_verified_ids, self.accept_length.sum().item()

    def filter_batch(self, new_indices: torch.Tensor):
        pass

    def merge_batch(self, spec_info: LookaheadVerifyInput):
        pass