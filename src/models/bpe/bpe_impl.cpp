// models/bpe/bpe_impl.cpp — BPE 实现
#include "models/bpe/bpe_impl.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "utils/unicode.h"

namespace bpe {

namespace {

// 构造 byte_fallback 的 token 字符串 "<0xAB>"(大写 2 位十六进制)
std::string byte_token(uint8_t b) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
    return std::string(buf);
}

}  // namespace

absl::StatusOr<std::unique_ptr<bpe_internal::Word>> BPE::Impl::merge_word(std::string_view text) const {
    auto word = std::make_unique<bpe_internal::Word>();
    if (text.empty()) {
        return word;
    }

    // 用于 fuse_unk:记录上一个 Symbol 是否是 UNK
    bool last_was_unk = false;

    // 遍历 UTF-8 码点
    util::codepoint_iter it(text);
    auto end_it = it.end();
    std::size_t char_idx = 0;
    const std::size_t total_chars = static_cast<std::size_t>(std::distance(it.begin(), end_it));

    for (auto v : it) {
        const bool is_first = (char_idx == 0);
        const bool is_last = (char_idx + 1 == total_chars);
        ++char_idx;

        // 构造 vocab 查找键:prefix + char_bytes + suffix
        std::string key;
        if (!is_first && opts.continuing_subword_prefix) {
            key += *opts.continuing_subword_prefix;
        }
        key.append(v.bytes.data(), v.bytes.size());
        if (is_last && opts.end_of_word_suffix) {
            key += *opts.end_of_word_suffix;
        }

        auto found = vocab.find(key);
        if (found != vocab.end()) {
            word->add(found->second, static_cast<uint32_t>(v.bytes.size()));
            last_was_unk = false;
            continue;
        }

        // 未命中 vocab
        if (opts.byte_fallback) {
            // 逐字节发 <0xXX> token
            for (std::size_t bi = 0; bi < v.bytes.size(); ++bi) {
                auto b = static_cast<uint8_t>(v.bytes[bi]);
                std::string btok = byte_token(b);
                auto bit = vocab.find(btok);
                if (bit != vocab.end()) {
                    word->add(bit->second, 1u);
                } else if (opts.unk_token) {
                    // byte token 自身不在 vocab,退化为 UNK
                    auto uit = vocab.find(*opts.unk_token);
                    if (uit == vocab.end()) {
                        return absl::InternalError("unk_token not in vocab but byte_fallback needed it");
                    }
                    word->add(uit->second, 1u);
                } else {
                    return absl::InvalidArgumentError(
                        absl::StrCat("byte token '", btok, "' not in vocab and no unk_token"));
                }
            }
            last_was_unk = false;
            continue;
        }

        // 非 byte_fallback:发 UNK
        if (opts.unk_token) {
            auto uit = vocab.find(*opts.unk_token);
            if (uit == vocab.end()) {
                return absl::InternalError("unk_token not in vocab");
            }
            TokenId unk_id = uit->second;
            if (opts.fuse_unk && last_was_unk) {
                // 与上一个相邻 UNK 合并:不新增 symbol,扩展上一个活跃 symbol 的字节长度
                word->extend_last(static_cast<uint32_t>(v.bytes.size()));
            } else {
                word->add(unk_id, static_cast<uint32_t>(v.bytes.size()));
            }
            last_was_unk = true;
            continue;
        }

        return absl::InvalidArgumentError(
            absl::StrCat("character '", std::string(v.bytes), "' not in vocab and no unk_token/byte_fallback"));
    }

    // 执行合并
    word->merge_all(merges, opts.dropout);
    return word;
}

std::vector<Token> BPE::Impl::word_to_tokens(const bpe_internal::Word& w) const {
    auto ids_ = w.ids();
    auto offs = w.offsets();
    std::vector<Token> out;
    out.reserve(ids_.size());
    for (std::size_t i = 0; i < ids_.size(); ++i) {
        Token t;
        t.id = ids_[i];
        t.offsets = offs[i];
        auto it = vocab_r.find(t.id);
        t.value = (it != vocab_r.end()) ? it->second : std::string{};
        out.push_back(std::move(t));
    }
    return out;
}

std::vector<Token> BPE::Impl::tokenize(std::string_view text) const {
    if (text.empty()) {
        return {};
    }

    // 1. ignore_merges 整词短路(GPT-2 高频词)
    if (opts.ignore_merges) {
        auto it = vocab.find(text);
        if (it != vocab.end()) {
            Token t;
            t.id = it->second;
            t.value = std::string(text);
            t.offsets = {0, static_cast<uint32_t>(text.size())};
            return {std::move(t)};
        }
    }

    // 2. 缓存命中
    if (cache) {
        auto cached = cache->get(text);
        if (cached) {
            return word_to_tokens(*cached);
        }
    }

    // 3. merge_word
    auto result = merge_word(text);
    if (!result.ok()) {
        return {};  // M2:简化错误处理,返回空
    }

    auto word = std::move(*result);

    // 4. 缓存插入
    if (cache && text.size() <= BPE::max_cached_length()) {
        cache->put(std::string(text), std::make_unique<bpe_internal::Word>(*word));
    }

    return word_to_tokens(*word);
}

// ---- BPE 类方法(pImpl 转发)-------------------------------------------------
BPE::BPE(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}

BPE::~BPE() = default;
BPE::BPE(BPE&&) noexcept = default;
BPE& BPE::operator=(BPE&&) noexcept = default;

std::vector<Token> BPE::tokenize(std::string_view text) const {
    return impl_->tokenize(text);
}

std::optional<TokenId> BPE::token_to_id(std::string_view t) const {
    auto it = impl_->vocab.find(t);
    if (it == impl_->vocab.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::optional<std::string> BPE::id_to_token(TokenId id) const {
    auto it = impl_->vocab_r.find(id);
    if (it == impl_->vocab_r.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::size_t BPE::vocab_size() const {
    return impl_->vocab.size();
}

const BpeOptions& BPE::options() const noexcept {
    return impl_->opts;
}

const Vocab& BPE::vocab() const noexcept {
    return impl_->vocab;
}

MergeList BPE::merges_list() const {
    // 把 MergeMap(无序)转为按 rank 排序的 MergeList
    // 每条 (Pair{a,b}, {rank, merged_id}) → (vocab_r[a], vocab_r[b])
    std::vector<std::pair<uint32_t, std::pair<std::string, std::string>>> ranked;
    ranked.reserve(impl_->merges.size());
    for (const auto& [pair, mv] : impl_->merges) {
        auto ia = impl_->vocab_r.find(pair.a);
        auto ib = impl_->vocab_r.find(pair.b);
        if (ia == impl_->vocab_r.end() || ib == impl_->vocab_r.end()) {
            continue;
        }
        ranked.emplace_back(mv.rank, std::make_pair(ia->second, ib->second));
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    MergeList out;
    out.reserve(ranked.size());
    for (auto& [_, pair] : ranked) {
        out.push_back(std::move(pair));
    }
    return out;
}

void BPE::clear_cache() {
    if (impl_->cache) {
        impl_->cache->clear();
    }
}

}  // namespace bpe
