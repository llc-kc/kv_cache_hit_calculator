# kv_cache_hit_calculator
Calculate the theoretical hit rate of requests



example usage:

```
usage example
python kimi_cache_hit_simulator.py \
  --input req_10.jsonl \
  --output req_10_cache_hits.jsonl \
  --model-path /data1/models/Kimi-K3 \
  --page-size 64 \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3 \
  --trust-remote-code
```



输入：

包含请求内容的jsonl字符串文件。

输出：

理论命中率大小, example:
```
request_id="9a4c19ae-fcb4-4df9-ac20-820393d97b97" num_tokens=5269 cache_hit_tokens=4736 cache_hit_rate=89.88%
request_id="e696faa9-3114-4017-86ab-77e387a9a270" num_tokens=5771 cache_hit_tokens=4864 cache_hit_rate=84.28%
request_id="be94a676-ae56-40d6-894c-a42b1f8174ab" num_tokens=4656 cache_hit_tokens=4224 cache_hit_rate=90.72%
request_id="9d48d698-42fb-4e5d-ab2a-fb0d31408138" num_tokens=5698 cache_hit_tokens=4480 cache_hit_rate=78.62%
Converted 1000 request(s); 0 failed. Request hit: 992/1000 requests (99.20%); cache hit: 8417792/11575414 prompt tokens (72.72%); 8417792/11543360 cacheable tokens (72.92%); 48837 unique page hashes.
```
