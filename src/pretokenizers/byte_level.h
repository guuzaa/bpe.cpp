// pretokenizers/byte_level.h — GPT-2 ByteLevel:PreTokenizer + Decoder + PostProcessor
//
// 设计参考 §7.1 / HF tokenizers byte_level.rs。
// 三合一职责:
//   PreTokenizer:
//     1. (可选) 若 add_prefix_space 且字符串不以空格开头,prepend " "
//     2. (可选) 用 GPT-2 正则切分
//     3. 把每个字节的 UTF-8 序列映射为 byte-char(如空格 0x20 → U+0120 'Ġ')
//   Decoder:把 byte-char 序列还原为字节,再 from_utf8_lossy 拼成字符串
//   PostProcessor:trim_offsets — 修剪 token 偏移中的前后导空白
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "bpe/interfaces.h"
#include "bpe/types.h"
#include "pretokenizers/regex_engine.h"

namespace bpe::pretokenizers {

class ByteLevel : public PreTokenizer, public Decoder, public PostProcessor {
   public:
    ByteLevel(bool add_prefix_space = true, bool trim_offsets = true, bool use_regex = true);

    // ---- PreTokenizer ------------------------------------------------------
    void pre_tokenize(PreTokenizedString& s) const override;

    // ---- Decoder -----------------------------------------------------------
    std::string decode(const std::vector<std::string>& tokens) const override;

    // ---- PostProcessor -----------------------------------------------------
    std::vector<Token> process(std::vector<Token> tokens) const override;

    // ---- 配置 --------------------------------------------------------------
    bool add_prefix_space() const noexcept {
        return add_prefix_space_;
    }

    bool trim_offsets() const noexcept {
        return trim_offsets_;
    }

    bool use_regex() const noexcept {
        return use_regex_;
    }

    // 把一段普通字符串字节映射为 byte-char 字符串(供测试与外部使用)
    static std::string encode_bytes(std::string_view s);
    // 把 byte-char 字符串还原为字节流(失败字节按 0xEF 0xBF 0xBD 替换)
    static std::string decode_bytes(std::string_view s);

   private:
    // 修剪单个 token 的前后导空白偏移(对应 HF process_offsets)
    void trim_token_offsets(Token& t, std::size_t token_index) const;

    bool add_prefix_space_;
    bool trim_offsets_;
    bool use_regex_;
    Regex regex_;  // GPT-2 正则
};

}  // namespace bpe::pretokenizers
