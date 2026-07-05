// bpe/trainer.h — BpeTrainer 公共 API
//
// 对应设计文档 §6 / HF tokenizers trainer.rs。
// 用法:
//   auto trainer = BpeTrainer::builder()
//                     .vocab_size(1000)
//                     .min_frequency(2)
//                     .build();
//   trainer.feed(input_strings, *pretokenizer);
//   auto status = trainer.train(*bpe_model);
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "bpe/bpe.h"
#include "bpe/interfaces.h"

namespace bpe {

struct BpeTrainerOptions {
    uint64_t min_frequency = 2;                                           // pair 出现次数下限
    std::size_t vocab_size = 30000;                                       // 目标词表大小
    std::vector<std::string> special_tokens;                              // 预置 token( 如 <unk>)
    std::optional<std::size_t> limit_alphabet;                            // 字母表裁剪上限
    std::vector<std::string> initial_alphabet;                            // 强制纳入的字符
    std::optional<std::string> continuing_subword_prefix;                 // 如 "##"
    std::optional<std::string> end_of_word_suffix;                        // 如 "</w>"
    std::optional<std::size_t> max_token_length;                          // 单 token 字节长度上限
    std::function<void(std::size_t, std::size_t, const char*)> progress;  // (cur, total, stage)
};

class BpeTrainer {
   public:
    BpeTrainer();
    explicit BpeTrainer(BpeTrainerOptions opts);
    ~BpeTrainer();
    BpeTrainer(BpeTrainer&&) noexcept;
    BpeTrainer& operator=(BpeTrainer&&) noexcept;

    // 链式 builder
    static BpeTrainer builder_like(BpeTrainerOptions opts);

    // 累积语料:对每条字符串应用 pretokenizer,把每个 pre-token 当作词,
    // 累计词频到内部 word_counts。可重复调用以分批输入。
    void feed(const std::vector<std::string>& inputs, const PreTokenizer& pretokenizer);

    // 训练:把累积的 word_counts 训出 merges/vocab,写回 *model。
    // 训练会清空并重建 model 的 vocab/vocab_r/merges(保留 opts.special_tokens)。
    absl::Status train(BPE& model);

    std::size_t word_count() const noexcept;

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace bpe