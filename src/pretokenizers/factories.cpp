// src/pretokenizers/factories.cpp — PreTokenizer 工厂实现
#include <utility>

#include "bpe/pretokenizers.h"
#include "pretokenizers/byte_level.h"
#include "pretokenizers/sequence.h"
#include "pretokenizers/split.h"

namespace bpe {

std::unique_ptr<PreTokenizer> make_byte_level_pretokenizer(bool add_prefix_space, bool trim_offsets, bool use_regex) {
    return std::make_unique<pretokenizers::ByteLevel>(add_prefix_space, trim_offsets, use_regex);
}

SplitBehavior parse_split_behavior(std::string_view s) noexcept {
    if (s == "Removed") {
        return SplitBehavior::kRemoved;
    }
    if (s == "Isolated") {
        return SplitBehavior::kIsolated;
    }
    if (s == "MergedWithPrevious") {
        return SplitBehavior::kMergedWithPrevious;
    }
    if (s == "MergedWithNext") {
        return SplitBehavior::kMergedWithNext;
    }
    if (s == "Contiguous") {
        return SplitBehavior::kContiguous;
    }
    return SplitBehavior::kIsolated;
}

std::unique_ptr<PreTokenizer> make_split_pretokenizer(std::string pattern, SplitPatternKind kind,
                                                      SplitBehavior behavior, bool invert) {
    return std::make_unique<pretokenizers::Split>(std::move(pattern), kind, behavior, invert);
}

std::unique_ptr<PreTokenizer> make_sequence_pretokenizer(std::vector<std::unique_ptr<PreTokenizer>> pretokenizers) {
    // pretokenizers::Sequence 持有 vector<unique_ptr<PreTokenizer>>,直接搬运基类指针。
    return std::make_unique<pretokenizers::Sequence>(std::move(pretokenizers));
}

}  // namespace bpe