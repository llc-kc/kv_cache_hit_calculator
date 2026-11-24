use std::collections::HashMap;
use std::env;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

const DEFAULT_INPUT: &str = "dsv41.1000.evalscope.token_ids.jsonl";
const DEFAULT_OUTPUT: &str = "prefix_matches.jsonl";
const DEFAULT_PAGE_SIZE: usize = 256;
const BUFFER_SIZE: usize = 1024 * 1024;

struct Config {
    input: PathBuf,
    output: PathBuf,
    page_size: usize,
}

impl Config {
    fn parse() -> Result<Option<Self>, String> {
        let mut config = Self {
            input: DEFAULT_INPUT.into(),
            output: DEFAULT_OUTPUT.into(),
            page_size: DEFAULT_PAGE_SIZE,
        };
        let mut args = env::args_os().skip(1);
        while let Some(arg) = args.next() {
            let value = |arg: &OsString, value: Option<OsString>| {
                value.ok_or_else(|| format!("missing value for {}", arg.to_string_lossy()))
            };
            match arg.to_str() {
                Some("--input" | "-i") => config.input = value(&arg, args.next())?.into(),
                Some("--output" | "-o") => config.output = value(&arg, args.next())?.into(),
                Some("--page-size" | "-p") => {
                    config.page_size = value(&arg, args.next())?
                        .to_str()
                        .and_then(|value| value.parse::<NonZeroUsize>().ok())
                        .ok_or("page size must be a positive integer")?
                        .get();
                }
                Some("--help" | "-h") => return Ok(None),
                _ => return Err(format!("unknown argument {}", arg.to_string_lossy())),
            }
        }
        Ok(Some(config))
    }
}

struct Node {
    first_request: NonZeroUsize,
    next_collision: Option<NonZeroUsize>,
}

struct PrefixIndex {
    page_size: usize,
    children: HashMap<(usize, u64), Option<NonZeroUsize>>,
    nodes: Vec<Node>,
    pages: Vec<u32>,
}

impl PrefixIndex {
    fn new(page_size: usize) -> Self {
        Self {
            page_size,
            children: HashMap::new(),
            nodes: Vec::new(),
            pages: Vec::new(),
        }
    }

    fn insert(&mut self, request: NonZeroUsize, tokens: &[u32]) -> (usize, Option<NonZeroUsize>) {
        let mut parent = 0;
        let mut matched_tokens = 0;
        let mut matched_request = None;
        for page in tokens.chunks_exact(self.page_size) {
            let fingerprint = page.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, &token| {
                (hash ^ u64::from(token)).wrapping_mul(0x0100_0000_01b3)
            });
            let head = self.children.entry((parent, fingerprint)).or_default();
            let mut candidate = *head;
            while let Some(id) = candidate {
                let index = id.get() - 1;
                let start = index * self.page_size;
                let node = &self.nodes[index];
                if self.pages[start..start + self.page_size] == *page {
                    matched_tokens += self.page_size;
                    matched_request = Some(node.first_request);
                    break;
                }
                candidate = node.next_collision;
            }
            if let Some(id) = candidate {
                parent = id.get();
            } else {
                // A new parent has no descendants, so later pages cannot extend the match.
                let id = NonZeroUsize::new(self.nodes.len() + 1).unwrap();
                self.nodes.push(Node {
                    first_request: request,
                    next_collision: *head,
                });
                self.pages.extend_from_slice(page);
                *head = Some(id);
                parent = id.get();
            }
        }
        (matched_tokens, matched_request)
    }
}

fn skip_whitespace(mut input: &[u8]) -> &[u8] {
    while matches!(input.first(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
        input = &input[1..];
    }
    input
}

fn parse_tokens(mut input: &[u8], tokens: &mut Vec<u32>) -> Result<(), &'static str> {
    tokens.clear();
    input = skip_whitespace(input)
        .strip_prefix(b"[")
        .ok_or("expected an array of unsigned token IDs")?;
    input = skip_whitespace(input);
    if !input.starts_with(b"]") {
        loop {
            let mut digits = 0;
            let mut value = 0_u64;
            for &byte in input {
                if !byte.is_ascii_digit() {
                    break;
                }
                if digits == 10 || (digits == 1 && value == 0) {
                    return Err("invalid unsigned 32-bit token ID");
                }
                value = value * 10 + u64::from(byte - b'0');
                digits += 1;
            }
            if digits == 0 || value > u64::from(u32::MAX) {
                return Err("expected an unsigned 32-bit token ID");
            }
            tokens.push(value as u32);
            input = skip_whitespace(&input[digits..]);
            match input.first() {
                Some(b',') => input = skip_whitespace(&input[1..]),
                Some(b']') => break,
                _ => return Err("expected ',' or ']' after token ID"),
            }
        }
    }
    if !skip_whitespace(&input[1..]).is_empty() {
        return Err("unexpected data after token array");
    }
    Ok(())
}

struct RequestMatch {
    predecessor: Option<NonZeroUsize>,
    hit_rate: f64,
}

fn process(input: File, output: File, page_size: usize) -> io::Result<usize> {
    let mut reader = BufReader::with_capacity(BUFFER_SIZE, input);
    let mut writer = BufWriter::with_capacity(BUFFER_SIZE, output);
    let mut index = PrefixIndex::new(page_size);
    let mut line = Vec::new();
    let mut tokens = Vec::new();
    let mut history: Vec<RequestMatch> = Vec::new();

    while reader.read_until(b'\n', &mut line)? != 0 {
        let request = NonZeroUsize::new(history.len() + 1).unwrap();
        parse_tokens(&line, &mut tokens).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("line {request}: {error}"),
            )
        })?;
        let (matched_tokens, matched_request) = index.insert(request, &tokens);
        let hit_rate = if tokens.is_empty() {
            0.0
        } else {
            matched_tokens as f64 / tokens.len() as f64
        };
        write!(writer, "{{\"request_id\":{request},\"matched_request_id\":")?;
        match matched_request {
            Some(id) => write!(writer, "{id}")?,
            None => writer.write_all(b"null")?,
        }
        write!(
            writer,
            ",\"matched_tokens\":{matched_tokens},\"hit_rate\":{hit_rate:.2},\"chain\":\"{request}"
        )?;
        let mut predecessor = matched_request;
        let mut chain_length = 1;
        while let Some(id) = predecessor {
            write!(writer, "->{id}")?;
            chain_length += 1;
            predecessor = history[id.get() - 1].predecessor;
        }
        write!(
            writer,
            "\",\"chain_length\":{chain_length},\"chain_hit_rate\":\""
        )?;
        if let Some(id) = matched_request {
            write!(writer, "{hit_rate:.2}")?;
            let mut previous = &history[id.get() - 1];
            while let Some(id) = previous.predecessor {
                write!(writer, "->{:.2}", previous.hit_rate)?;
                previous = &history[id.get() - 1];
            }
        }
        writer.write_all(b"\"}\n")?;
        history.push(RequestMatch {
            predecessor: matched_request,
            hit_rate,
        });
        line.clear();
    }
    writer.flush()?;
    Ok(history.len())
}

fn run(config: &Config) -> Result<usize, String> {
    let input_path = fs::canonicalize(&config.input)
        .map_err(|error| format!("{}: {error}", config.input.display()))?;
    if fs::canonicalize(&config.output).is_ok_and(|output| output == input_path) {
        return Err("input and output must be different files".into());
    }
    let input =
        File::open(&input_path).map_err(|error| format!("{}: {error}", config.input.display()))?;
    let temporary = config
        .output
        .with_file_name(format!(".prefix-match-{}.tmp", std::process::id()));
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| format!("{}: {error}", temporary.display()))?;
    let result = process(input, output, config.page_size)
        .map_err(|error| {
            format!(
                "{} → {}: {error}",
                config.input.display(),
                config.output.display()
            )
        })
        .and_then(|count| {
            fs::rename(&temporary, &config.output)
                .map_err(|error| format!("{}: {error}", config.output.display()))?;
            Ok(count)
        });
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn print_help() {
    println!(
        "Usage: prefix-match [OPTIONS]\n\n\
         -i, --input PATH       Token ID arrays in JSONL [default: {DEFAULT_INPUT}]\n\
         -o, --output PATH      Prefix matches in JSONL [default: {DEFAULT_OUTPUT}]\n\
         -p, --page-size N      Tokens per complete page [default: {DEFAULT_PAGE_SIZE}]\n\
         -h, --help             Show this help\n\n\
         Request IDs start at 1. Ties go to the earliest matching request.\n\
         Hit rate is matched tokens divided by total request tokens (0 for empty requests).\n\
         Chain length counts all requests in the chain, including the current request.\n\
         Chain hit rate lists the rate for each link in order, or an empty string if there are no links.\n\
         All hit rates are written with two decimal places.\n\
         Example output: {{\"request_id\":100,\"matched_request_id\":80,\"matched_tokens\":768,\"hit_rate\":0.75,\"chain\":\"100->80->50\",\"chain_length\":3,\"chain_hit_rate\":\"0.75->0.50\"}}"
    );
}

fn main() -> ExitCode {
    match Config::parse() {
        Ok(None) => print_help(),
        Ok(Some(config)) => {
            let start = Instant::now();
            match run(&config) {
                Ok(count) => eprintln!(
                    "Processed {count} requests in {:.3}s → {}",
                    start.elapsed().as_secs_f64(),
                    config.output.display()
                ),
                Err(error) => {
                    eprintln!("Error: {error}");
                    return ExitCode::FAILURE;
                }
            }
        }
        Err(error) => {
            eprintln!("Error: {error}. See --help.");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}
