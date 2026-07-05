// models/bpe/trainer_impl.h — BpeTrainer 内部实现
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "bpe/trainer.h"
#include "bpe/types.h"
#include "models/bpe/serialization.h"  // MergeMap / MergeValue
#include "models/bpe/word.h"

namespace bpe {

using bpe_internal::MergeMap;
using bpe_internal::MergeValue;
using bpe_internal::Word;

// 训练堆条目:{pair, count, 出现的 word 索引集合}
struct TrainMerge {
    Pair pair;
    uint64_t count;
    absl::flat_hash_set<uint32_t> positions;

    // 用作 d_ary_heap 的比较:count 大者优先(pair 小者打破并列)
    bool operator<(const TrainMerge& o) const noexcept {
        if (count != o.count) {
            return count < o.count;
        }
        return o.pair < pair;  // pair 小者优先(反向)
    }
};

// pair 计数 + word 索引集合
struct PairStats {
    int64_t count = 0;
    absl::flat_hash_set<uint32_t> positions;
};

class BpeTrainer::Impl {
   public:
    explicit Impl(bpe::BpeTrainerOptions opts) : opts_(std::move(opts)) {
    }

    void feed(const std::vector<std::string>& inputs, const PreTokenizer& pretokenizer);

    absl::Status train(BPE& model);

    std::size_t word_count() const noexcept {
        return word_counts_.size();
    }

   private:
    // 1. 把 special_tokens 加入 vocab
    void add_special_tokens(absl::flat_hash_map<std::string, TokenId>& word_to_id,
                            std::vector<std::string>& id_to_word);

    // 2. 计算字母表,写入 vocab;返回是否需要 prefix/suffix
    void compute_alphabet(absl::flat_hash_map<std::string, TokenId>& word_to_id, std::vector<std::string>& id_to_word);

    // 3. 把 word_counts 展开为 Vec<Word> + counts
    std::pair<std::vector<std::unique_ptr<Word>>, std::vector<uint64_t>> tokenize_words(
        absl::flat_hash_map<std::string, TokenId>& word_to_id, std::vector<std::string>& id_to_word);

    // 4. 并行 pair 计数
    std::unordered_map<Pair, PairStats, PairHash> count_pairs(const std::vector<std::unique_ptr<Word>>& words,
                                                              const std::vector<uint64_t>& counts);

    // 5. 合并主循环(惰性 stale-count + 并行 apply merge)
    void merge_loop(std::vector<std::unique_ptr<Word>>& words, const std::vector<uint64_t>& counts,
                    std::unordered_map<Pair, PairStats, PairHash>& pair_counts,
                    absl::flat_hash_map<std::string, TokenId>& word_to_id, std::vector<std::string>& id_to_word,
                    std::vector<std::pair<Pair, TokenId>>& merges_out);

    BpeTrainerOptions opts_;
    absl::flat_hash_map<std::string, uint64_t> word_counts_;
};

}  // namespace bpe