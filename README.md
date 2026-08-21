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

理论命中率大小。

