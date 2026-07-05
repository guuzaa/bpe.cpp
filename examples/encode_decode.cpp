// examples/encode_decode.cpp — 加载 tokenizer.json,对输入文本做 encode + decode
//
// 用法:
//   ./encode_decode                              # 用内置示例文本
//   ./encode_decode "Hello world"                # 自定义文本
//   ./encode_decode "Hello world" assets/deepseek-v3-2-tokenizer.json
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "bpe/tokenizer.h"
#include "utils/bytes_char.h"
#include "utils/unicode.h"

int main(int argc, char** argv) {
    std::string config = "assets/deepseek-v3-2-tokenizer.json";
    std::string_view text = "Hello, world! This is bpe.cpp.";

    if (argc >= 2) {
        text = argv[1];
    }
    if (argc >= 3) {
        config = argv[2];
    }

    std::cerr << "Loading tokenizer: " << config << " ...\n";
    auto t0 = std::chrono::steady_clock::now();
    auto status = bpe::Tokenizer::from_file(config);
    auto t1 = std::chrono::steady_clock::now();
    if (!status.ok()) {
        std::cerr << "Failed to load: " << status.status().message() << "\n";
        return 1;
    }
    auto& tok = *status.value();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cerr << "Loaded in " << load_ms << " ms"
              << " (vocab_size=" << tok.model()->vocab_size() << ")\n\n";

    // ---- Encode ----------------------------------------------------------
    std::cerr << "Input: \"" << text << "\"\n";
    auto t2 = std::chrono::steady_clock::now();
    bpe::Encoding enc = tok.encode(text);
    auto t3 = std::chrono::steady_clock::now();
    auto enc_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::printf("Encode: %zu tokens in %lld us\n", enc.ids.size(), enc_us);
    for (std::size_t i = 0; i < enc.ids.size(); ++i) {
        // token value 是 byte-char 字符串(每个 codepoint 对应一个原始字节)
        // 先按 codepoint 迭代解码回原始字节,再做转义显示
        std::string raw_bytes;
        for (auto v : bpe::util::codepoint_iter(enc.tokens[i])) {
            int b = bpe::util::char_to_byte(v.cp);
            if (b >= 0) {
                raw_bytes.push_back(static_cast<char>(b));
            } else {
                // 非 byte-char 字符(special token 等),保留原字节
                raw_bytes.append(v.bytes.data(), v.bytes.size());
            }
        }
        // 转义显示原始字节
        std::string display;
        for (char ch : raw_bytes) {
            auto c = static_cast<unsigned char>(ch);
            if (c >= 0x20 && c < 0x7f) {
                display.push_back(static_cast<char>(c));
            } else {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\x%02X", c);
                display += buf;
            }
        }
        std::printf("  [%3zu] id=%-7u  token=%-24s  bytes=%zu  offsets=[%u,%u)\n", i, enc.ids[i], display.c_str(),
                    static_cast<std::size_t>(enc.offsets[i].second - enc.offsets[i].first), enc.offsets[i].first,
                    enc.offsets[i].second);
    }

    // ---- Decode ----------------------------------------------------------
    auto t4 = std::chrono::steady_clock::now();
    std::string recovered = tok.decode(enc.ids);
    auto t5 = std::chrono::steady_clock::now();
    auto dec_us = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();

    std::printf("\nDecode: \"%s\" in %lld us\n", recovered.c_str(), dec_us);
    if (recovered == text) {
        std::cerr << "Round-trip: OK\n";
    } else {
        std::cerr << "Round-trip: MISMATCH\n";
        std::cerr << "  expected: \"" << text << "\"\n";
        std::cerr << "  got:      \"" << recovered << "\"\n";
    }

    return recovered == text ? 0 : 1;
}
