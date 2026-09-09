import ast
import hashlib
import struct
import heapq

import numpy as np
import sys


def compute_prefix_hash(prefix_hash: int, input_ids: list[int]) -> int:
    h = hashlib.blake2b(digest_size=16)

    if prefix_hash is not None:
        h.update(prefix_hash.to_bytes(16, byteorder="little", signed=False))

    # 加入当前 token ids
    for token_id in input_ids:
        h.update(struct.pack("<Q", token_id))

    # 返回 128-bit int hash
    return int.from_bytes(h.digest(), byteorder="little")


def get_prefix_hash_list(list_ids, page_size=64):
    in_len = len(list_ids)
    hash_list = []
    prefix_hash = None
    for start in range(0, in_len, page_size):
        prefix_hash = compute_prefix_hash(prefix_hash, list_ids[start:start+page_size])
        hash_list.append(prefix_hash)
    return hash_list


"""
the input ids in the file should be like:
[1, 2, 3]
[4, 5, 6]
"""

input_file = sys.argv[1]

out_file = input_file + ".full_hit.log"

page_size = 64
topk = 3

input_ids_all: list[list[int]] = []

with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue

        input_ids = ast.literal_eval(line)
        input_ids_all.append(input_ids)

print(f"load input_ids_all finish, num : {len(input_ids_all)}")

def get_topk(data, k):
    topk = heapq.nlargest(k, enumerate(data), key=lambda x: x[1])
    values = [x[1] for x in topk]
    indices = [x[0] for x in topk]
    return values, indices


req_hashes = [set(get_prefix_hash_list(input_ids, page_size)) for input_ids in input_ids_all]
print("compute req_hashes finish")



# req_mamba_store_pos = []

with open(out_file, "w") as f:
    for i, list_ids in enumerate(input_ids_all):
        prev_match_nums = []
        for j in range(0, i):
            prefix_num = len(req_hashes[i] & req_hashes[j])
            prev_match_nums.append(prefix_num)

        hit_pages, indices = get_topk(prev_match_nums, topk)
        cur_len = len(list_ids)
        cur_len_aligned = cur_len//page_size*page_size 
        hit_pages = np.array(hit_pages)
        hit_len = hit_pages * page_size
        hit_rate = hit_len / cur_len

        # # mamba hit logics
        # mamba_store_pos = cur_len_aligned
        # expected_mamba_hit = 0
        # if len(hit_len) > 0:
        #     max_hit_prev_id = indices[0]
        #     max_hit_len = hit_len[0]
        #     prev_mamba_pos = req_mamba_store_pos[max_hit_prev_id]
        #     if prev_mamba_pos >0 and prev_mamba_pos <= max_hit_len:
        #         expected_mamba_hit = prev_mamba_pos
        #         mamba_store_pos = cur_len_aligned
        #     else:
        #         expected_mamba_hit = 0
        #         mamba_store_pos = max_hit_len
        # req_mamba_store_pos.append(mamba_store_pos)

        cached_input_len_max = hit_len[0] if len(hit_len) else hit_len
        log_str = f"input_len={cur_len}, cached_input_len={cached_input_len_max}, cached_input_len topk={hit_len}, cached_prev_ids: {indices}, hit_rate: {hit_rate}"
        print(log_str)

        f.write(log_str + "\n")
        # if i > 10:
        #     break
