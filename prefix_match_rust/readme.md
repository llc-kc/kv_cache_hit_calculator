# Prefix Match

Prefix Match 是一个 Rust 命令行工具，用于分析一批 LLM 请求的前缀复用情况。它按输入顺序处理请求，为每个请求找出前缀匹配最长的历史请求，并输出命中长度、命中率和匹配链。

## 匹配规则

1. 将每个请求从第一个 token 开始切成固定大小的页，默认为 256 个 token。末尾不足一页的 token 不参与匹配，但计入请求总长度。
2. 从第一页开始比较连续的完整页。每页通过哈希定位，再比较实际 token 序列，以避免哈希碰撞造成误匹配。
3. 每条请求只匹配它前面的请求。多条历史请求的命中长度相同时，选择输入中最早出现的请求。
4. 沿每条请求记录的最佳匹配逐级回溯。例如 `96->58->38` 表示请求 96 的最佳匹配是 58，请求 58 的最佳匹配是 38。

工具逐行读取输入，通过前缀索引查找历史匹配。索引的内存占用随不同的完整页前缀数量增长。

## 输入格式

输入为 JSONL。每行是一个请求的 token ID 数组，token ID 为无符号 32 位整数。请求 ID 从 1 开始，对应输入文件的行号。空数组 `[]` 表示空请求。

```jsonl
[10,20,30,40]
[10,20,50,60]
[10,20,50,60,70]
```

## 使用方法

需要支持 Rust 2024 Edition 的 Rust 工具链和 Cargo。在项目目录执行下列命令，使用 Release 构建处理默认输入。

```powershell
cargo run --release
```

指定输入文件、输出文件和页大小。

```powershell
cargo run --release -- --input requests.jsonl --output matches.jsonl --page-size 2
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `-i`、`--input PATH` | 输入 JSONL 文件路径 | `dsv41.1000.evalscope.token_ids.jsonl` |
| `-o`、`--output PATH` | 输出 JSONL 文件路径 | `prefix_matches.jsonl` |
| `-p`、`--page-size N` | 每页 token 数，须为正整数 | `256` |
| `-h`、`--help` | 显示帮助 | |

相对路径以当前工作目录为准。运行成功后会替换同名输出文件。

## 输出格式

输出为 JSONL，每条请求一行，顺序与输入一致。把上面的示例输入保存为 `requests.jsonl`，按页大小 2 运行后，第三条结果如下。

```json
{"request_id":3,"matched_request_id":2,"matched_tokens":4,"hit_rate":0.80,"chain":"3->2->1","chain_length":3,"chain_hit_rate":"0.80->0.50"}
```

| 字段 | 含义 |
| --- | --- |
| `request_id` | 当前请求的行号，从 1 开始 |
| `matched_request_id` | 最佳匹配的历史请求 ID，无命中时为 `null` |
| `matched_tokens` | 命中页数乘以页大小，无命中时为 `0` |
| `hit_rate` | 命中 token 数除以当前请求的总 token 数，空请求记为 `0.00` |
| `chain` | 用 `->` 连接匹配链中的请求 ID，无命中时只包含当前请求 |
| `chain_length` | 链中的请求数，包含当前请求，无命中时为 `1` |
| `chain_hit_rate` | 用 `->` 连接各跳的命中率，无命中时为空字符串 |

两处命中率均按 0 到 1 的比例输出，保留两位小数。示例中 `3->2` 命中 4 个 token，请求 3 共 5 个 token，因此命中率为 `0.80`。`2->1` 命中 2 个 token，请求 2 共 4 个 token，因此命中率为 `0.50`。
