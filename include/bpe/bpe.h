// bpe/bpe.h — BPE 模型公共 API
//
// 对应设计文档 §4.5。BPE 是 Model 接口的具体实现:
//   - 持有 vocab / vocab_r / merges / options / cache
//   - tokenize(text) 走 merge_word + Word::merge_all
//   - 通过 BpeBuilder 链式构造
//
// 注意:本头只暴露对外 API,内部状态(Word/Cache/MergeMap 等)藏在 src/ 的 Impl 里,
// 通过 pImpl 减少 include/bpe/ 的依赖污染。
#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "bpe/interfaces.h"
#include "bpe/types.h"

namespace bpe {

struct BpeOptions {
    std::optional<std::string> unk_token;
    std::optional<std::string> continuing_subword_prefix;  // 如 "##"
    std::optional<std::string> end_of_word_suffix;         // 如 "</w>"
    bool fuse_unk = false;
    bool byte_fallback = false;
    bool ignore_merges = false;          // GPT-2 整词命中即返回
    std::optional<float> dropout;        // BPE-Dropout, 0/None = 确定性
    std::size_t cache_capacity = 10000;  // 0 = 关闭缓存
};

// 构建器:链式设置参数,build() 产出 BPE 实例。
// 设计参考 Rust 版 BpeBuilder(model.rs:108)。
class BpeBuilder {
   public:
    BpeBuilder();
    ~BpeBuilder();
    BpeBuilder(BpeBuilder&&) noexcept;
    BpeBuilder& operator=(BpeBuilder&&) noexcept;
    BpeBuilder(const BpeBuilder&) = delete;
    BpeBuilder& operator=(const BpeBuilder&) = delete;

    BpeBuilder& vocab_and_merges(Vocab vocab, MergeList merges);
    BpeBuilder& files(std::string vocab_json_path, std::string merges_txt_path);
    BpeBuilder& cache_capacity(std::size_t n);
    BpeBuilder& dropout(float p);
    BpeBuilder& unk_token(std::string t);
    BpeBuilder& continuing_subword_prefix(std::string p);
    BpeBuilder& end_of_word_suffix(std::string s);
    BpeBuilder& fuse_unk(bool v);
    BpeBuilder& byte_fallback(bool v);
    BpeBuilder& ignore_merges(bool v);

    // 构造 BPE;若设置了 files 则读取文件,若 vocab/merges 缺失返回 Status 错误
    absl::StatusOr<std::unique_ptr<class BPE>> build();

    // 便捷:一步构造空 BPE(空 vocab + 空 merges),等价于 .vocab_and_merges({},{}).build()
    // 用于训练场景:先建空模型,再由 BpeTrainer 填充 vocab/merges
    static absl::StatusOr<std::unique_ptr<class BPE>> build_default();

   private:
    struct ImplBuilder;
    std::unique_ptr<ImplBuilder> p_;
};

class BPE : public Model {
   public:
    ~BPE() override;
    BPE(const BPE&) = delete;
    BPE& operator=(const BPE&) = delete;
    BPE(BPE&&) noexcept;
    BPE& operator=(BPE&&) noexcept;

    // ---- Model 接口 --------------------------------------------------------
    std::vector<Token> tokenize(std::string_view text) const override;
    std::optional<TokenId> token_to_id(std::string_view t) const override;
    std::optional<std::string> id_to_token(TokenId id) const override;
    std::size_t vocab_size() const override;

    // ---- BPE 专属 ----------------------------------------------------------
    const BpeOptions& options() const noexcept;
    const Vocab& vocab() const noexcept;

    // 返回 merges 列表(按 rank 排序),用于序列化
    MergeList merges_list() const;
    void clear_cache();

    static std::size_t max_cached_length() noexcept {
        return 256;
    }

    // 仅供 BpeBuilder / BpeTrainer 使用
    struct Impl;
    explicit BPE(std::unique_ptr<Impl> impl);
    friend class BpeTrainer;

   private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace bpe
