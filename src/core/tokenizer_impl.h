// core/tokenizer_impl.h — Tokenizer 内部实现
#pragma once

#include <memory>
#include <vector>

#include "bpe/tokenizer.h"

namespace bpe {

struct Tokenizer::Impl {
    std::unique_ptr<Normalizer> normalizer;
    std::unique_ptr<PreTokenizer> pretokenizer;
    std::unique_ptr<Model> model;
    std::unique_ptr<PostProcessor> post_processor;
    std::unique_ptr<Decoder> decoder;
    std::optional<TruncationOptions> truncation;
    std::optional<PaddingOptions> padding;

    Encoding encode_one(std::string_view text, bool add_special_tokens) const;
    void apply_truncation(Encoding& e) const;
    void apply_padding(std::vector<Encoding>& batch) const;
};

}  // namespace bpe