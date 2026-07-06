// models/bpe/builder.cpp
#include "models/bpe/builder.h"
#include <utility>

#include "absl/status/status.h"
#include "bpe/bpe.h"
#include "models/bpe/bpe_impl.h"
#include "models/bpe/serialization.h"

namespace bpe {

BpeBuilder::BpeBuilder() : p_(std::make_unique<ImplBuilder>()) {
}

BpeBuilder::~BpeBuilder() = default;
BpeBuilder::BpeBuilder(BpeBuilder&&) noexcept = default;
BpeBuilder& BpeBuilder::operator=(BpeBuilder&&) noexcept = default;

BpeBuilder& BpeBuilder::vocab_and_merges(Vocab vocab, MergeList merges) {
    p_->vocab = std::move(vocab);
    p_->merges = std::move(merges);
    return *this;
}

BpeBuilder& BpeBuilder::files(std::string vocab_json_path, std::string merges_txt_path) {
    p_->vocab_json_path = std::move(vocab_json_path);
    p_->merges_txt_path = std::move(merges_txt_path);
    return *this;
}

BpeBuilder& BpeBuilder::cache_capacity(std::size_t n) {
    p_->opts.cache_capacity = n;
    return *this;
}

BpeBuilder& BpeBuilder::dropout(float p) {
    p_->opts.dropout = p;
    return *this;
}

BpeBuilder& BpeBuilder::unk_token(std::string t) {
    p_->opts.unk_token = std::move(t);
    return *this;
}

BpeBuilder& BpeBuilder::continuing_subword_prefix(std::string p) {
    p_->opts.continuing_subword_prefix = std::move(p);
    return *this;
}

BpeBuilder& BpeBuilder::end_of_word_suffix(std::string s) {
    p_->opts.end_of_word_suffix = std::move(s);
    return *this;
}

BpeBuilder& BpeBuilder::fuse_unk(bool v) {
    p_->opts.fuse_unk = v;
    return *this;
}

BpeBuilder& BpeBuilder::byte_fallback(bool v) {
    p_->opts.byte_fallback = v;
    return *this;
}

BpeBuilder& BpeBuilder::ignore_merges(bool v) {
    p_->opts.ignore_merges = v;
    return *this;
}

absl::StatusOr<std::unique_ptr<BPE>> BpeBuilder::build_default() {
    return BpeBuilder().vocab_and_merges({}, {}).build();
}

absl::StatusOr<std::unique_ptr<BPE>> BpeBuilder::build() {
    // 1. 读取文件(若设置了路径)
    if (p_->vocab_json_path) {
        auto v = bpe_internal::read_vocab_json(*p_->vocab_json_path);
        if (!v.ok()) {
            return v.status();
        }
        p_->vocab = *std::move(v);
    }
    if (p_->merges_txt_path) {
        auto m = bpe_internal::read_merges_txt(*p_->merges_txt_path);
        if (!m.ok()) {
            return m.status();
        }
        p_->merges = *std::move(m);
    }

    if (!p_->vocab) {
        return absl::InvalidArgumentError("vocab not provided");
    }
    if (!p_->merges) {
        p_->merges = MergeList{};  // 允许空 merges(只有单字符词表)
    }

    // 2. 构建 vocab_r(id -> token)
    BPE::Impl impl;
    impl.vocab = *std::move(p_->vocab);
    for (const auto& [tok, id] : impl.vocab) {
        impl.vocab_r[id] = tok;
    }

    // 3. 构建 MergeMap
    auto mm = bpe_internal::convert_merges_to_hashmap(impl.vocab, *p_->merges);
    if (!mm.ok()) {
        return mm.status();
    }
    impl.merges = *std::move(mm);

    // 4. 选项 + 缓存
    impl.opts = std::move(p_->opts);
    if (impl.opts.cache_capacity > 0) {
        impl.cache = std::make_unique<bpe_internal::BpeCache>(impl.opts.cache_capacity);
    }

    // 5. dropout 校验
    if (impl.opts.dropout && (*impl.opts.dropout < 0.0f || *impl.opts.dropout > 1.0f)) {
        return absl::InvalidArgumentError("dropout must be in [0, 1]");
    }

    return std::unique_ptr<BPE>(new BPE(std::make_unique<BPE::Impl>(std::move(impl))));
}

}  // namespace bpe
