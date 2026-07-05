// bpe/tokenizer.h — Tokenizer 总线
//
// 五阶段管道编排(对应设计 §2):
//   text → AddedVocabulary/Normalizer → PreTokenizer → Model::tokenize
//        → PostProcessor → Truncation/Padding → Encoding
//
// 各阶段通过基类指针持有,M6 阶段 AddedVocabulary 暂以"切出 special tokens"的最小
// 形态存在;Padding 后续扩展。
#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "bpe/decoders.h"
#include "bpe/interfaces.h"
#include "bpe/normalizers.h"
#include "bpe/pretokenizers.h"
#include "bpe/types.h"

namespace bpe {

struct TruncationOptions {
    std::optional<std::size_t> max_length;  // 保留的最大 token 数
    std::optional<std::size_t> stride;      // 滑窗步长(0=不滑窗)
    bool direction_from_end = false;        // 从尾部截
};

struct PaddingOptions {
    std::optional<std::size_t> max_length;  // 目标长度(可与 batch 最长对齐)
    TokenId pad_id = 0;
    std::string pad_token = "<pad>";
    TokenId pad_type_id = 0;
};

class Tokenizer {
   public:
    Tokenizer();
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    // ---- 组件 setter(链式)--------------------------------------------------
    Tokenizer& set_normalizer(std::unique_ptr<Normalizer> n);
    Tokenizer& set_pretokenizer(std::unique_ptr<PreTokenizer> p);
    Tokenizer& set_model(std::unique_ptr<Model> m);
    Tokenizer& set_post_processor(std::unique_ptr<PostProcessor> p);
    Tokenizer& set_decoder(std::unique_ptr<Decoder> d);
    Tokenizer& set_truncation(TruncationOptions opts);
    Tokenizer& set_padding(PaddingOptions opts);

    // ---- 组件 getter --------------------------------------------------------
    Model* model() const noexcept;
    const Normalizer* normalizer() const noexcept;
    Decoder* decoder() const noexcept;

    // ---- 编码 --------------------------------------------------------------
    // 对单条文本:返回 Encoding(ids/tokens/offsets/...)
    Encoding encode(std::string_view text, bool add_special_tokens = true) const;

    // 批量:多条文本;若开启 padding,统一 pad 到 batch 内最长(或 padding.max_length)
    std::vector<Encoding> encode_batch(const std::vector<std::string>& texts, bool add_special_tokens = true) const;

    // ---- 解码 --------------------------------------------------------------
    std::string decode(const std::vector<TokenId>& ids, bool skip_special_tokens = true) const;

    // ---- tokenizer.json / vocab.json+merges.txt 读写 ----------------------
    // 从单文件 tokenizer.json 装配出 Tokenizer(自动构造 BPE/ByteLevel/Decoder)
    static absl::StatusOr<std::unique_ptr<Tokenizer>> from_file(const std::string& path);

    // 把当前 Tokenizer 序列化为 tokenizer.json
    absl::Status to_file(const std::string& path) const;

    // 把 model 词表写为 vocab.json + merges.txt(M6 最小可用)
    absl::Status to_vocab_merges(const std::string& dir) const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace bpe