// models/bpe/builder.h — BpeBuilder::ImplBuilder(pImpl)
#pragma once

#include <optional>
#include <string>

#include "bpe/bpe.h"

namespace bpe {

struct BpeBuilder::ImplBuilder {
    std::optional<std::string> vocab_json_path;
    std::optional<std::string> merges_txt_path;
    std::optional<Vocab> vocab;
    std::optional<MergeList> merges;
    BpeOptions opts;
};

}  // namespace bpe
