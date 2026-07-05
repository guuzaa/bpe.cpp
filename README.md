# bpe.cpp

A C++17 implementation of Byte Pair Encoding (BPE) tokenization, inspired by HuggingFace `tokenizers` (Rust).

## Features

- **Encode / Decode / Train** — full BPE pipeline
- **GPT-2 byte-level** — regex pre-tokenization with PCRE2 (JIT), byte-char mapping
- **Multiple variants** — `</w>` (Sennrich), `##` (BERT), `byte_fallback` (SentencePiece)
- **Tokenizer.json** — load/save HuggingFace-compatible tokenizer files
- **Thread-safe cache** — `thread_local` LRU for encode hot path

## Dependencies

| Library | Purpose |
|---|---|
| [abseil-cpp](https://github.com/abseil/abseil-cpp) | `flat_hash_map`, `Status`/`StatusOr`, `Mutex` |
| [nlohmann/json](https://github.com/nlohmann/json) | JSON serialization |
| [PCRE2](https://github.com/PCRE2Project/pcre2) | Regex for pre-tokenization (UTF-8, UCP, JIT) |
| [googletest](https://github.com/google/googletest) | Tests only |

All fetched via CMake `FetchContent`; system PCRE2 preferred if available.

## Build

```bash
cmake --preset debug && cmake --build --preset debug
```

## Test

```bash
ctest --preset debug
```

## Format

```bash
cmake --build --preset debug -t format
```

## Usage

```cpp
#include "bpe/tokenizer.h"

auto tok = bpe::Tokenizer::from_file("tokenizer.json");
bpe::Encoding enc = tok->encode("Hello, world!");
std::string decoded = tok->decode(enc.ids);
```

See `examples/encode_decode.cpp` for a complete example.

## Architecture

```
raw text → Normalizer → PreTokenizer → Model::tokenize → PostProcessor → Encoding
                                                    ↓
                                              ids → Decoder → String
```

- **Normalizer** — strip, case, identity
- **PreTokenizer** — GPT-2 byte-level (PCRE2 regex), whitespace, punctuation
- **Model** — BPE with priority-queue merge + lazy invalidation
- **Decoder** — byte-level, byte-fallback

## License

See [LICENSE](./LICENSE) file.
