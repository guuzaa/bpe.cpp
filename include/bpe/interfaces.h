// bpe/interfaces.h — 五阶段管道各组件的抽象基类
//
// 设计参考 HuggingFace tokenizers 的 Rust trait:
//   Normalizer / PreTokenizer / Model / PostProcessor / Decoder / Trainer
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "bpe/types.h"

namespace bpe {

// 带偏移对齐的字符串容器
//
// 在 normalize 阶段会原地修改 normalized 文本,同时维护 normalized ↔ original 的字节
// 偏移对齐,以便后续 PreTokenizer/Model 产出的 token 偏移能反追到原文本。
// M6 实现最小版本:支持 append/replace/strip/lower/upper 等基本变换,
// 并提供 align_to_original() 把 normalized 字节偏移转回 original 字节偏移。
class NormalizedString {
   public:
    explicit NormalizedString(std::string s) : original_(std::move(s)), normalized_(original_) {
        // 初始:aligned_to_original_[i] = i,即每个 normalized 字节对应同一 original 字节
        aligned_to_original_.resize(normalized_.size());
        for (std::size_t i = 0; i < aligned_to_original_.size(); ++i) {
            aligned_to_original_[i] = static_cast<uint32_t>(i);
        }
    }

    std::string_view get() const noexcept {
        return normalized_;
    }

    std::string_view original() const noexcept {
        return original_;
    }

    // 把 normalized 字节偏移转为 original 字节偏移(左闭右开)
    std::pair<uint32_t, uint32_t> align_to_original(uint32_t norm_start, uint32_t norm_end) const;

    // M6 内部接口,供 normalizer 使用:原地替换 [norm_start, norm_end) 为 replacement
    void replace_range(uint32_t norm_start, uint32_t norm_end, std::string_view replacement);

    // 在末尾追加(M6 不用)
    void append(std::string_view s);

    // 把整段 normalized 转小写 / 转大写(ASCII + Latin-1)
    void lowercase();
    void uppercase();

    // 去首尾空白字节
    void strip();

   private:
    std::string original_;
    std::string normalized_;
    // aligned_to_original_[i] = normalized 第 i 字节对应的 original 字节偏移
    std::vector<uint32_t> aligned_to_original_;
};

// 预切分得到的一段文本(及其在原字符串中的偏移)
struct PreToken {
    std::string text;
    std::pair<uint32_t, uint32_t> offsets;  // 对齐到 normalize 后的字节
    // alignment[i] = text 第 i 字节对应的原始段内字节偏移
    // 空 = 无需映射(text 字节偏移 == 原始段字节偏移,如非 ByteLevel 的简单 pre-tokenizer)
    // ByteLevel 会填充此项,因为 byte-char 编码后 text 长度 ≠ 原始段长度
    std::vector<uint32_t> alignment;
};

class PreTokenizedString {
   public:
    explicit PreTokenizedString(std::vector<PreToken> toks) : pretokens_(std::move(toks)) {
    }

    const std::vector<PreToken>& tokens() const noexcept {
        return pretokens_;
    }

    std::vector<PreToken>& tokens() noexcept {
        return pretokens_;
    }

   private:
    std::vector<PreToken> pretokens_;
};

class Normalizer {
   public:
    virtual ~Normalizer() = default;
    virtual void normalize(NormalizedString& s) const = 0;
};

class PreTokenizer {
   public:
    virtual ~PreTokenizer() = default;
    virtual void pre_tokenize(PreTokenizedString& s) const = 0;
};

class Model {
   public:
    virtual ~Model() = default;
    // 对一段 pre-token 子串做子词切分
    virtual std::vector<Token> tokenize(std::string_view text) const = 0;
    virtual std::optional<TokenId> token_to_id(std::string_view t) const = 0;
    virtual std::optional<std::string> id_to_token(TokenId id) const = 0;
    virtual std::size_t vocab_size() const = 0;
};

class PostProcessor {
   public:
    virtual ~PostProcessor() = default;
    virtual std::vector<Token> process(std::vector<Token> tokens) const = 0;
};

class Decoder {
   public:
    virtual ~Decoder() = default;
    virtual std::string decode(const std::vector<std::string>& tokens) const = 0;
};

class Trainer {
   public:
    virtual ~Trainer() = default;
};

}  // namespace bpe
