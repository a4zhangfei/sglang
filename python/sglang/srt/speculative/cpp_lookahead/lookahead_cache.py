# -*- coding: utf-8 -*-

# from sglang.op.lookahead import Lookahead, Param

import logging
import os

import numpy as np
from torch.utils.cpp_extension import load

logger = logging.getLogger(__name__)

_abs_path = os.path.dirname(os.path.abspath(__file__))
lookahead_cache_cpp = load(
    name="lookahead_cache_cpp",
    sources=[
        f"{_abs_path}/lookahead_cache_binding.cpp",
        f"{_abs_path}/lookahead.cpp",
    ],
    extra_cflags=["-O3", "-std=c++20"],
    extra_ldflags=["-lglog"],
)

class LookaheadCache:
    def __init__(
        self,
        branch_length=18,
        min_match_window_size=1,
        max_match_window_size=10,
        min_bfs_breadth=1,
        max_bfs_breadth=7,
        return_token_limit=7,
        capacity=1000000,
    ):
        param = lookahead_cache_cpp.Param()
        param.branch_length = branch_length
        param.min_match_window_size = min_match_window_size
        param.max_match_window_size = max_match_window_size
        param.min_bfs_breadth = min_bfs_breadth
        param.max_bfs_breadth = max_bfs_breadth
        param.return_token_limit = return_token_limit - 1
        param.capacity = capacity
        self.cache = lookahead_cache_cpp.Lookahead(capacity, param)
        
        self.default_mask = np.ones((1, 1), dtype=np.int64)
        self.return_token_limit = return_token_limit

    def put(self, token_ids):
        self.cache.async_insert(token_ids)
        
    def synchronize(self):
        self.cache.synchronize()
        
    def reset(self):
        self.cache.reset()

    def get(
        self,
        token_ids,
        batch_size=1,
    ):
        # return token_ids[-1:], self.default_mask
        result = self.cache.matchBFS(token_ids, batch_size)
        # result = self.cache.matchProb(token_ids, batch_size)
        if len(result.token) == 0:
            result.token, result.mask = [11], [1]
            # return token_ids[-1:], self.default_mask

        # 优化1: 使用numpy数组操作避免list拷贝
        result_tokens = np.asarray(result.token)
        
        # 新增逻辑：当匹配的token数量不足时，补充随机token
        original_mask = np.asarray(result.mask, dtype=np.int64).reshape(len(result.token), -1)
        target_token_num = self.return_token_limit - 1
        if len(result_tokens) < target_token_num:
            # 计算需要补充的token数量
            tokens_to_add = target_token_num - len(result_tokens)
            # 生成随机tokens (范围0-10000，可根据实际词汇表大小调整)
            random_tokens = np.random.randint(0, 10000, size=tokens_to_add, dtype=result_tokens.dtype)
            # 拼接原始tokens和随机tokens
            result_tokens = np.concatenate([result_tokens, random_tokens])
            
            # 扩展mask：原始mask + 对角线扩展
            original_rows, original_cols = original_mask.shape
            # 创建扩展后的mask
            extended_mask = np.zeros((target_token_num, max(target_token_num, original_cols)), dtype=original_mask.dtype)
            # 拷贝原始mask到左上角
            extended_mask[:original_rows, :original_cols] = original_mask
            # 为新增的tokens设置对角线为1（表示每个新token只依赖于之前的所有tokens）
            for i in range(original_rows, target_token_num):
                if i < extended_mask.shape[1]:
                    extended_mask[i, i] = 1
            original_mask = extended_mask
            
        last_token = token_ids[-1]
        # 预分配数组并直接填充，避免拼接操作
        modified_tokens = np.empty(len(result_tokens) + 1, dtype=result_tokens.dtype)
        modified_tokens[0] = last_token
        modified_tokens[1:] = result_tokens
        rows, cols = original_mask.shape

        new_mask = np.zeros((rows + 1, cols + 1), dtype=original_mask.dtype)
        new_mask[1:, 1:] = original_mask
        new_mask[:, 0] = 1

        return modified_tokens.tolist(), new_mask
    
    def generate_all_valid_paths(self, tokens, tree_mask):
        """
        根据tokens和tree_mask，返回每一个token可见的token列表（即每一行的可见token索引）。
        Args:
            tokens: List[int]，token序列
            tree_mask: np.ndarray，形状为(n_tokens, n_tokens)，mask[i][j]=1表示第i个token可见第j个token
        Returns:
            List[List[int]]，每个token可见的token索引列表
        """
        if len(tokens) == 0 or tree_mask.size == 0:
            return []
        n_tokens = len(tokens)
        result = []
        for i in range(n_tokens):
            visible_indices = [j for j in range(n_tokens) if tree_mask[i][j] == 1]
            result.append([tokens[j] for j in visible_indices])
        return result
        

# main function
if __name__ == "__main__":
    format = f"%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    token_ids=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 44, 55, 66, 77, 88, 99, 100]]
    cache = LookaheadCache(branch_length=12, return_token_limit=8)
    for token_id in token_ids:
        cache.put(token_id)

    cache.synchronize()
    decoding_ids, decoding_masks = cache.get(token_ids=[1, 2, 3])
    logger.info(f"{decoding_ids=}, decoding_masks=\n{decoding_masks}")

    leaf_paths = cache.generate_all_valid_paths(decoding_ids, decoding_masks)
    for i, path in enumerate(leaf_paths):
        logger.info(f"draft path {i}: {path}")