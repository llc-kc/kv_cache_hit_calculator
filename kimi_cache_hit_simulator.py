"""Render, tokenize, and simulate prefix-cache hits for Kimi K3 requests.

Each input line is expected to look like this::

    {"content": "{\"messages\":[...],\"model\":\"kimi-k3\"}", ...}

``content`` may also already be a JSON object.  The output is JSONL with the
outer log metadata, the rendered model prompt, the prompt token IDs, and the
token count.  When ``--page-size`` is supplied, requests are also processed in
file order by an unbounded global prefix-hash cache.  Only the tokenizer is
loaded; this script does not load model weights or start an SGLang server.

The rendering path deliberately calls ``OpenAIServingChat._process_messages``.
That method dispatches through ``_apply_jinja_template`` and Kimi K3's custom
``_encode_messages`` branch, keeping this offline conversion aligned with the
SGLang serving path.
"""

"""
usage example
python kimi_cache_hit_simulator.py \
  --input req_10.jsonl \
  --output req_10_cache_hits.jsonl \
  --model-path /data1/models/Kimi-K3 \
  --page-size 64 \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3 \
  --trust-remote-code
"""

# from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Sequence, TextIO


class InputLineError(ValueError):
    """An invalid JSONL record, annotated with its source line number."""


def compute_page_hashes(token_ids: Sequence[int], page_size: int) -> list[str]:
    """Return SGLang's chained hashes for complete token pages.

    Prefix cache keys are page-aligned, so an incomplete final page is
    deliberately excluded before calling SGLang's canonical hash function.
    """

    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")

    cacheable_tokens = len(token_ids) // page_size * page_size
    aligned_token_ids = list(token_ids[:cacheable_tokens])

    # Keep this import local so tokenize-only mode and ``--help`` do not load
    # SGLang's full memory-cache dependency graph.
    from sglang.srt.mem_cache.utils import get_hash_str

    page_hashes = get_hash_str(aligned_token_ids, page_size=page_size)
    if not isinstance(page_hashes, list):
        raise TypeError(
            "SGLang get_hash_str returned a single hash when page hashes "
            "were requested"
        )
    return page_hashes


@dataclass(frozen=True)
class CacheEntry:
    """Provenance for the first request that populated a page hash."""

    request_id: Any
    page_number: int


class PrefixHashCache:
    """An unbounded global dictionary of position-aware KV page hashes."""

    def __init__(self, page_size: int) -> None:
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")
        self.page_size = page_size
        self.entries: dict[str, CacheEntry] = {}

    def lookup(
        self, page_hashes: Iterable[str]
    ) -> tuple[int, CacheEntry | None]:
        """Return the cached prefix length and its furthest-page provenance."""

        matched_pages = 0
        source_entry = None
        for page_hash in page_hashes:
            entry = self.entries.get(page_hash)
            if entry is None:
                break
            matched_pages += 1
            source_entry = entry
        return matched_pages, source_entry

    def insert(self, page_hashes: Iterable[str], request_id: Any) -> None:
        """Make all of a completed request's page hashes available later."""

        for page_number, page_hash in enumerate(page_hashes, start=1):
            self.entries.setdefault(
                page_hash,
                CacheEntry(request_id=request_id, page_number=page_number),
            )

    def query_then_insert(
        self, token_ids: Sequence[int], request_id: Any
    ) -> tuple[list[str], int, Any | None]:
        """Query before insertion and return hashes, hit tokens, and source ID."""

        page_hashes = compute_page_hashes(token_ids, self.page_size)
        matched_pages, source_entry = self.lookup(page_hashes)
        self.insert(page_hashes, request_id)
        source_request_id = (
            source_entry.request_id if source_entry is not None else None
        )
        return page_hashes, matched_pages * self.page_size, source_request_id


class _OfflineModelConfig:
    """The subset of ModelConfig used by OpenAIServingChat initialization."""

    is_multimodal = True
    hf_config = SimpleNamespace(
        architectures=["KimiK3ForConditionalGeneration"],
        model_type="kimi_k3",
    )

    @staticmethod
    def get_default_sampling_params() -> dict[str, Any]:
        return {}


class KimiK3RequestTokenizer:
    """CPU-only adapter around SGLang's serving-time request renderer."""

    def __init__(
        self,
        model_path: str,
        *,
        tool_call_parser: str,
        reasoning_parser: str,
        tokenizer_mode: str = "auto",
        tokenizer_revision: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        # Keep SGLang imports local: ``--help`` and JSON error reporting do not
        # need to import its full serving dependency graph.
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
        from sglang.srt.parser.template_manager import TemplateManager
        from sglang.srt.utils.hf_transformers import get_tokenizer

        tokenizer = get_tokenizer(
            model_path,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
        )
        server_args = SimpleNamespace(
            default_chat_template_kwargs=None,
            reasoning_parser=reasoning_parser,
            tool_call_parser=tool_call_parser,
        )
        tokenizer_manager = SimpleNamespace(
            model_config=_OfflineModelConfig(),
            served_model_name=model_path,
            server_args=server_args,
            tokenizer=tokenizer,
        )

        self.tokenizer = tokenizer
        self.serving = OpenAIServingChat(tokenizer_manager, TemplateManager())
        if self.serving.chat_encoding_spec != "kimi_k3":
            raise RuntimeError(
                "SGLang did not select the Kimi K3 chat encoder; pass "
                "--tool-call-parser kimi_k3 and use a Kimi K3 tokenizer."
            )

    def tokenize(self, request_body: dict[str, Any]) -> tuple[str, list[int]]:
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        request = ChatCompletionRequest.model_validate(request_body)
        result = self.serving._process_messages(
            request,
            # Kimi K3 is a VLM, and the online serving path passes the model's
            # multimodal capability here even for text-only requests.
            is_multimodal=True,
        )
        input_ids = [int(token_id) for token_id in result.prompt_ids]

        # In the multimodal K3 branch, _apply_jinja_template decodes the exact
        # token IDs into ``result.prompt``.  Keep that serving-produced string
        # instead of independently recreating the chat template here.
        prompt = result.prompt
        if not isinstance(prompt, str):
            raise TypeError(
                f"SGLang returned prompt type {type(prompt).__name__}, expected str"
            )
        return prompt, input_ids


def _parse_input_line(
    raw_line: str, line_number: int, content_field: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        outer = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise InputLineError(
            f"line {line_number}: invalid outer JSON: {exc.msg}"
        ) from exc
    if not isinstance(outer, dict):
        raise InputLineError(
            f"line {line_number}: outer JSON must be an object, got "
            f"{type(outer).__name__}"
        )
    if content_field not in outer:
        raise InputLineError(
            f"line {line_number}: missing request field {content_field!r}"
        )

    content = outer[content_field]
    if isinstance(content, str):
        try:
            request_body = json.loads(content)
        except json.JSONDecodeError as exc:
            raise InputLineError(
                f"line {line_number}: {content_field!r} is not valid JSON: {exc.msg}"
            ) from exc
    elif isinstance(content, dict):
        request_body = content
    else:
        raise InputLineError(
            f"line {line_number}: {content_field!r} must be a JSON string or "
            f"object, got {type(content).__name__}"
        )

    if not isinstance(request_body, dict):
        raise InputLineError(
            f"line {line_number}: decoded request body must be an object, got "
            f"{type(request_body).__name__}"
        )
    metadata = {key: value for key, value in outer.items() if key != content_field}
    return metadata, request_body


def _iter_records(stream: TextIO) -> Iterator[tuple[int, str]]:
    for line_number, raw_line in enumerate(stream, start=1):
        if raw_line.strip():
            yield line_number, raw_line


def _open_input(path: str):
    if path == "-":
        return nullcontext(sys.stdin)
    return Path(path).open("r", encoding="utf-8-sig")


def _open_output(path: str):
    if path == "-":
        return nullcontext(sys.stdout)
    return Path(path).open("w", encoding="utf-8", newline="\n")


def _write_jsonl(stream: TextIO, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    stream.write("\n")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize JSONL-logged Kimi K3 Chat Completions requests and "
            "optionally simulate SGLang-compatible prefix-cache hits."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL path, or '-' for stdin.",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output JSONL path (default: stdout).",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local Kimi K3 model/tokenizer directory or Hugging Face repo ID.",
    )
    parser.add_argument(
        "--tool-call-parser",
        required=True,
        choices=("kimi_k3",),
        help="SGLang tool-call parser; Kimi K3 requires 'kimi_k3'.",
    )
    parser.add_argument(
        "--reasoning-parser",
        required=True,
        choices=("kimi_k3",),
        help="SGLang reasoning parser; Kimi K3 requires 'kimi_k3'.",
    )
    parser.add_argument(
        "--content-field",
        default="content",
        help="Outer JSON field containing the serialized request (default: content).",
    )
    parser.add_argument(
        "--request-id-field",
        default="request_id",
        help=(
            "Outer JSON field used as the request ID when simulating "
            "(default: request_id; falls back to the input line number)."
        ),
    )
    parser.add_argument(
        "--req-num",
        type=_positive_int,
        help="Only read the first N non-empty request records (default: all).",
    )
    parser.add_argument(
        "--page-size",
        type=_positive_int,
        help=(
            "Enable prefix-cache simulation with this page size. If omitted, "
            "the script retains its tokenize-only behavior."
        ),
    )
    parser.add_argument(
        "--include-page-hashes",
        action="store_true",
        help="Include the chained page hash keys in each output record.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print each request ID and cache-hit length to stderr.",
    )
    parser.add_argument(
        "--tokenizer-mode",
        choices=("auto", "slow"),
        default="auto",
        help="Tokenizer loading mode (default: auto).",
    )
    parser.add_argument(
        "--tokenizer-revision",
        help="Optional tokenizer revision.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow tokenizer code supplied by the model repository.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write an error record and continue instead of stopping at a bad line.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input != "-" and args.output != "-":
        if Path(args.input).resolve() == Path(args.output).resolve():
            raise SystemExit("--input and --output must be different files")
    if args.include_page_hashes and args.page_size is None:
        raise SystemExit("--include-page-hashes requires --page-size")

    prefix_cache = (
        PrefixHashCache(args.page_size) if args.page_size is not None else None
    )

    converter = KimiK3RequestTokenizer(
        args.model_path,
        tool_call_parser=args.tool_call_parser,
        reasoning_parser=args.reasoning_parser,
        tokenizer_mode=args.tokenizer_mode,
        tokenizer_revision=args.tokenizer_revision,
        trust_remote_code=args.trust_remote_code,
    )

    converted = 0
    failed = 0
    total_prompt_tokens = 0
    total_cacheable_tokens = 0
    total_cache_hit_tokens = 0
    total_cache_hit_requests = 0
    with _open_input(args.input) as input_stream, _open_output(
        args.output
    ) as output_stream:
        records: Iterable[tuple[int, str]] = _iter_records(input_stream)
        if args.req_num is not None:
            records = islice(records, args.req_num)
        for line_number, raw_line in records:
            try:
                metadata, request_body = _parse_input_line(
                    raw_line, line_number, args.content_field
                )
                prompt, input_ids = converter.tokenize(request_body)
                output_record = {
                    "line_number": line_number,
                    "metadata": metadata,
                    "prompt": prompt,
                    "input_ids": input_ids,
                    "num_tokens": len(input_ids),
                }

                if prefix_cache is not None:
                    request_id = metadata.get(args.request_id_field)
                    if request_id is None:
                        request_id = line_number
                    (
                        page_hashes,
                        cache_hit_tokens,
                        cache_hit_source_request_id,
                    ) = prefix_cache.query_then_insert(input_ids, request_id)
                    cacheable_tokens = len(page_hashes) * prefix_cache.page_size
                    request_cache_hit_rate = (
                        cache_hit_tokens / len(input_ids) if input_ids else 0.0
                    )
                    output_record.update(
                        {
                            "request_id": request_id,
                            "page_size": prefix_cache.page_size,
                            "num_pages": len(page_hashes),
                            "cacheable_tokens": cacheable_tokens,
                            "cache_hit_pages": (
                                cache_hit_tokens // prefix_cache.page_size
                            ),
                            "cache_hit_tokens": cache_hit_tokens,
                            "cache_hit_rate": request_cache_hit_rate,
                            "cache_hit_source_request_id": (
                                cache_hit_source_request_id
                            ),
                        }
                    )
                    if args.include_page_hashes:
                        output_record["page_hashes"] = page_hashes
                    if not args.quiet:
                        request_id_json = json.dumps(
                            request_id, ensure_ascii=False, separators=(",", ":")
                        )
                        source_request_id_json = json.dumps(
                            cache_hit_source_request_id,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        print(
                            f"request_id={request_id_json} "
                            f"num_tokens={len(input_ids)} "
                            f"cache_hit_tokens={cache_hit_tokens} "
                            f"cache_hit_rate={request_cache_hit_rate:.2%} "
                            f"cache_hit_source_request_id="
                            f"{source_request_id_json}",
                            file=sys.stderr,
                        )

                    total_cacheable_tokens += cacheable_tokens
                    total_cache_hit_tokens += cache_hit_tokens
                    if cache_hit_tokens > 0:
                        total_cache_hit_requests += 1

                _write_jsonl(output_stream, output_record)
                total_prompt_tokens += len(input_ids)
                converted += 1
            except Exception as exc:
                failed += 1
                if not args.continue_on_error:
                    if isinstance(exc, InputLineError):
                        raise
                    raise InputLineError(f"line {line_number}: {exc}") from exc
                _write_jsonl(
                    output_stream,
                    {
                        "line_number": line_number,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )

    summary = f"Converted {converted} request(s); {failed} failed."
    if prefix_cache is not None:
        prompt_hit_rate = (
            total_cache_hit_tokens / total_prompt_tokens
            if total_prompt_tokens
            else 0.0
        )
        cacheable_hit_rate = (
            total_cache_hit_tokens / total_cacheable_tokens
            if total_cacheable_tokens
            else 0.0
        )
        request_hit_rate = (
            total_cache_hit_requests / converted if converted else 0.0
        )
        summary += (
            f" Request hit: {total_cache_hit_requests}/{converted} requests "
            f"({request_hit_rate:.2%}); "
            f"cache hit: {total_cache_hit_tokens}/{total_prompt_tokens} "
            f"prompt tokens ({prompt_hit_rate:.2%}); "
            f"{total_cache_hit_tokens}/{total_cacheable_tokens} cacheable "
            f"tokens ({cacheable_hit_rate:.2%}); "
            f"{len(prefix_cache.entries)} unique page hashes."
        )
    print(summary, file=sys.stderr)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
