from typing import List

from .index_sharder import IndexSharder

class UniformIndexSharder(IndexSharder):
    def __init__(self, *, drop_extra_indexes: bool) -> None:
        self.drop_extra_indexes = drop_extra_indexes
        
    def __call__(self, *, n_indexes: int, n_shards: int) -> List[List[int]]:
        indexes = list(range(n_indexes))
        base_shard_len = n_indexes // n_shards
        extra_indexes = n_indexes % n_shards
        shards = []
        cur_shard = []
        for i in indexes:
            cur_shard.append(i)
            if self.drop_extra_indexes:
                if len(cur_shard) == base_shard_len:
                    shards.append(cur_shard.copy())
                    cur_shard = []
            else:
                if len(cur_shard) == base_shard_len and extra_indexes == 0:
                    shards.append(cur_shard.copy())
                    cur_shard = []
                elif len(cur_shard) == base_shard_len + 1:
                    shards.append(cur_shard.copy())
                    cur_shard = []
                    extra_indexes -= 1
        return shards