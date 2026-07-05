// models/bpe/bpe_impl.h — BPE 内部实现(pImpl)
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "bpe/bpe.h"
#include "bpe/types.h"
#include "models/bpe/cache.h"
#include "models/bpe/serialization.h"
#include "models/bpe/word.h"

namespace bpe {

// BPE::Impl 的定义(pImpl)
struct BPE::Impl {
    Vocab vocab;                    // token -> id
    VocabR vocab_r;                 // id -> token(build 时重建)
    bpe_internal::MergeMap merges;  // (a,b) -> (rank, merged_id)
    BpeOptions opts;
    std::unique_ptr<bpe_internal::BpeCache> cache;  // nullptr when capacity == 0

    // 核心:对单个 pre-token 做字符级展开 + 合并
    absl::StatusOr<std::unique_ptr<bpe_internal::Word>> merge_word(std::string_view text) const;

    // Word → Token 序列(用 vocab_r 填 value,用 Word::offsets 填 offsets)
    std::vector<Token> word_to_tokens(const bpe_internal::Word& w) const;

    // 主入口
    std::vector<Token> tokenize(std::string_view text) const;
};

}  // namespace bpe
