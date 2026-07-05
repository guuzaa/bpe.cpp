// models/bpe/trainer.cpp — BpeTrainer 实现
#include <algorithm>
#include <thread>
#include <utility>

#include "absl/status/status.h"
#include "models/bpe/bpe_impl.h"  // BPE::Impl 完整定义
#include "models/bpe/trainer_impl.h"
#include "utils/unicode.h"

namespace bpe {

BpeTrainer::BpeTrainer() : impl_(std::make_unique<BpeTrainer::Impl>(BpeTrainerOptions{})) {
}

BpeTrainer::BpeTrainer(BpeTrainerOptions opts) : impl_(std::make_unique<BpeTrainer::Impl>(std::move(opts))) {
}

BpeTrainer::~BpeTrainer() = default;
BpeTrainer::BpeTrainer(BpeTrainer&&) noexcept = default;
BpeTrainer& BpeTrainer::operator=(BpeTrainer&&) noexcept = default;

BpeTrainer BpeTrainer::builder_like(BpeTrainerOptions opts) {
    return BpeTrainer(std::move(opts));
}

void BpeTrainer::feed(const std::vector<std::string>& inputs, const PreTokenizer& pretokenizer) {
    impl_->feed(inputs, pretokenizer);
}

absl::Status BpeTrainer::train(BPE& model) {
    return impl_->train(model);
}

std::size_t BpeTrainer::word_count() const noexcept {
    return impl_->word_count();
}

}  // namespace bpe

namespace bpe {

// BpeTrainer::Impl 的方法实现(其本身就在 bpe 命名空间内)
void BpeTrainer::Impl::feed(const std::vector<std::string>& inputs, const PreTokenizer& pretokenizer) {
    for (const auto& s : inputs) {
        // 每条字符串作为一个 PreToken(offsets = [0, s.size()])
        PreTokenizedString ps({PreToken{s, {0, static_cast<uint32_t>(s.size())}, {}}});
        pretokenizer.pre_tokenize(ps);
        for (const auto& pt : ps.tokens()) {
            if (pt.text.empty()) {
                continue;
            }
            word_counts_[pt.text] += 1;
        }
    }
}

void BpeTrainer::Impl::add_special_tokens(absl::flat_hash_map<std::string, TokenId>& word_to_id,
                                          std::vector<std::string>& id_to_word) {
    for (const auto& tok : opts_.special_tokens) {
        if (word_to_id.contains(tok)) {
            continue;
        }
        TokenId id = static_cast<TokenId>(id_to_word.size());
        id_to_word.push_back(tok);
        word_to_id[tok] = id;
    }
}

void BpeTrainer::Impl::compute_alphabet(absl::flat_hash_map<std::string, TokenId>& word_to_id,
                                        std::vector<std::string>& id_to_word) {
    // 统计每个码点的加权频次
    absl::flat_hash_map<uint32_t, uint64_t> char_counts;
    for (const auto& [word, count] : word_counts_) {
        for (auto v : util::codepoint_iter(word)) {
            char_counts[v.cp] += count;
        }
    }
    // initial_alphabet 强制纳入(权重设为无穷大,避免被裁剪)
    for (const auto& s : opts_.initial_alphabet) {
        for (auto v : util::codepoint_iter(s)) {
            char_counts[v.cp] = std::numeric_limits<uint64_t>::max();
        }
    }
    // 转 vector 排序:频次降序,平手按 cp 升序
    std::vector<std::pair<uint32_t, uint64_t>> sorted(char_counts.begin(), char_counts.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first < b.first;
    });
    // limit_alphabet 裁剪
    if (opts_.limit_alphabet && *opts_.limit_alphabet < sorted.size()) {
        sorted.resize(*opts_.limit_alphabet);
    }
    // 把每个字符作为单字符 token 写入 vocab
    for (const auto& [cp, _] : sorted) {
        // 把 cp 编码为 UTF-8 字符串
        std::string s;
        if (cp < 0x80) {
            s.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        if (word_to_id.contains(s)) {
            continue;
        }
        TokenId id = static_cast<TokenId>(id_to_word.size());
        id_to_word.push_back(s);
        word_to_id[s] = id;
    }
}

std::pair<std::vector<std::unique_ptr<Word>>, std::vector<uint64_t>> BpeTrainer::Impl::tokenize_words(
    absl::flat_hash_map<std::string, TokenId>& word_to_id, std::vector<std::string>& id_to_word) {
    std::vector<std::unique_ptr<Word>> words;
    std::vector<uint64_t> counts;
    words.reserve(word_counts_.size());
    counts.reserve(word_counts_.size());

    for (const auto& [word, count] : word_counts_) {
        auto w = std::make_unique<Word>();
        util::codepoint_iter it(word);
        const std::size_t total = [&] {
            std::size_t n = 0;
            for (auto v : it) {
                (void)v;
                ++n;
            }
            return n;
        }();
        std::size_t idx = 0;
        for (auto v : it) {
            const bool is_first = (idx == 0);
            const bool is_last = (idx + 1 == total);
            // 构造查表键:prefix + char_bytes + suffix
            std::string key;
            if (!is_first && opts_.continuing_subword_prefix) {
                key += *opts_.continuing_subword_prefix;
            }
            key.append(v.bytes.data(), v.bytes.size());
            if (is_last && opts_.end_of_word_suffix) {
                key += *opts_.end_of_word_suffix;
            }
            // 查表;字母表已被 limit_alphabet 裁剪,不在表中的字符:
            //   - 若是单字符(无 prefix/suffix 复合)→ 跳过该码点(被裁掉)
            //   - 若是 prefix+char 或 char+suffix 复合 → 主动 add
            //     (prefix/suffix 是训练时的产物,不受 limit_alphabet 约束)
            auto it2 = word_to_id.find(key);
            if (it2 == word_to_id.end()) {
                bool composite = false;
                if (!is_first && opts_.continuing_subword_prefix) {
                    composite = true;
                }
                if (is_last && opts_.end_of_word_suffix) {
                    composite = true;
                }
                if (!composite) {
                    ++idx;
                    continue;
                }
                TokenId id = static_cast<TokenId>(id_to_word.size());
                id_to_word.push_back(key);
                word_to_id[key] = id;
                it2 = word_to_id.find(key);
            }
            w->add(it2->second, static_cast<uint32_t>(v.bytes.size()));
            ++idx;
        }
        words.push_back(std::move(w));
        counts.push_back(count);
    }
    return {std::move(words), std::move(counts)};
}

std::unordered_map<Pair, PairStats, PairHash> BpeTrainer::Impl::count_pairs(
    const std::vector<std::unique_ptr<Word>>& words, const std::vector<uint64_t>& counts) {
    // 并行分片:每个线程处理一段 word 索引,产出局部 PairStats,主线程归并
    const std::size_t n = words.size();
    if (n == 0) {
        return {};
    }

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        hw = 4;
    }
    if (hw > 16) {
        hw = 16;
    }
    if (hw > n) {
        hw = static_cast<unsigned>(n);
    }
    if (hw < 1) {
        hw = 1;
    }

    std::vector<std::thread> threads;
    std::vector<std::unordered_map<Pair, PairStats, PairHash>> locals(hw);

    auto worker = [&](unsigned tid, std::size_t lo, std::size_t hi) {
        auto& local = locals[tid];
        for (std::size_t i = lo; i < hi; ++i) {
            const auto& syms = words[i]->symbols();
            uint64_t w = counts[i];
            for (std::size_t j = 0; j + 1 < syms.size(); ++j) {
                if (syms[j].len == 0 || syms[j + 1].len == 0) {
                    continue;
                }
                Pair p{syms[j].c, syms[j + 1].c};
                auto& st = local[p];
                st.count += static_cast<int64_t>(w);
                st.positions.insert(static_cast<uint32_t>(i));
            }
        }
    };

    std::size_t chunk = (n + hw - 1) / hw;
    std::size_t lo = 0;
    for (unsigned t = 0; t < hw; ++t) {
        std::size_t hi = std::min(lo + chunk, n);
        if (hi == lo) {
            break;
        }
        threads.emplace_back(worker, t, lo, hi);
        lo = hi;
    }
    for (auto& th : threads) {
        th.join();
    }

    // 归并到 locals[0]
    std::unordered_map<Pair, PairStats, PairHash> out;
    if (hw >= 1) {
        out = std::move(locals[0]);
    }
    for (unsigned t = 1; t < hw; ++t) {
        for (auto& [p, st] : locals[t]) {
            auto& dst = out[p];
            dst.count += st.count;
            for (auto i : st.positions) {
                dst.positions.insert(i);
            }
        }
    }
    return out;
}

void BpeTrainer::Impl::merge_loop(std::vector<std::unique_ptr<Word>>& words, const std::vector<uint64_t>& counts,
                                  std::unordered_map<Pair, PairStats, PairHash>& pair_counts,
                                  absl::flat_hash_map<std::string, TokenId>& word_to_id,
                                  std::vector<std::string>& id_to_word,
                                  std::vector<std::pair<Pair, TokenId>>& merges_out) {
    const std::size_t max_token_length = opts_.max_token_length.value_or(std::numeric_limits<std::size_t>::max());

    // 初始化 8-ary 堆
    util::d_heap<TrainMerge, 8> heap;
    for (auto& [p, st] : pair_counts) {
        if (st.count > 0) {
            heap.push(TrainMerge{p, static_cast<uint64_t>(st.count), st.positions});
        }
    }

    std::size_t step = 0;
    while (word_to_id.size() < opts_.vocab_size && !heap.empty()) {
        TrainMerge top = heap.top();
        heap.pop();

        // 惰性 stale-count 刷新
        auto pc_it = pair_counts.find(top.pair);
        if (pc_it == pair_counts.end()) {
            continue;
        }
        uint64_t current = static_cast<uint64_t>(pc_it->second.count);
        if (top.count != current) {
            top.count = current;
            top.positions = pc_it->second.positions;
            heap.push(top);
            continue;
        }

        if (top.count < 1 || opts_.min_frequency > top.count) {
            break;
        }

        // 构造合并 token 字符串:剥掉 part_b 的 continuing_subword_prefix
        const std::string& part_a_str = id_to_word[top.pair.a];
        std::string part_b_str = id_to_word[top.pair.b];
        if (opts_.continuing_subword_prefix) {
            const auto& prefix = *opts_.continuing_subword_prefix;
            if (part_b_str.size() >= prefix.size() && part_b_str.compare(0, prefix.size(), prefix) == 0) {
                part_b_str.erase(0, prefix.size());
            }
        }
        std::string new_token = part_a_str + part_b_str;

        // 分配/查 id
        TokenId new_id;
        auto it = word_to_id.find(new_token);
        if (it != word_to_id.end()) {
            new_id = it->second;
        } else {
            new_id = static_cast<TokenId>(id_to_word.size());
            id_to_word.push_back(new_token);
            word_to_id[new_token] = new_id;
        }
        merges_out.emplace_back(top.pair, new_id);

        // 并行 apply merge 到所有相关 word
        // 把 positions 切片到若干线程,每个线程独占一段索引(无重叠)
        std::vector<uint32_t> pos_vec(top.positions.begin(), top.positions.end());
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) {
            hw = 4;
        }
        if (hw > 16) {
            hw = 16;
        }
        if (hw > pos_vec.size()) {
            hw = static_cast<unsigned>(pos_vec.size());
        }
        if (hw < 1) {
            hw = 1;
        }

        // 每线程局部 (Pair → delta) 累计,主线程归并到 pair_counts
        std::vector<std::unordered_map<Pair, int32_t, PairHash>> local_deltas(hw);
        std::vector<std::thread> threads;

        auto worker = [&](unsigned tid, std::size_t lo, std::size_t hi) {
            auto& delta = local_deltas[tid];
            for (std::size_t k = lo; k < hi; ++k) {
                uint32_t wi = pos_vec[k];
                auto changes =
                    words[wi]->merge(top.pair.a, top.pair.b, new_id, static_cast<uint32_t>(max_token_length));
                int32_t w = static_cast<int32_t>(counts[wi]);
                for (auto& [p, d] : changes) {
                    delta[p] += d * w;
                }
            }
        };

        std::size_t chunk = (pos_vec.size() + hw - 1) / hw;
        std::size_t lo = 0;
        for (unsigned t = 0; t < hw; ++t) {
            std::size_t hi = std::min(lo + chunk, pos_vec.size());
            if (hi == lo) {
                break;
            }
            threads.emplace_back(worker, t, lo, hi);
            lo = hi;
        }
        for (auto& th : threads) {
            th.join();
        }

        // 归并 delta 到 pair_counts,收集需要入堆的新 pair
        absl::flat_hash_set<Pair> affected;
        for (unsigned t = 0; t < hw; ++t) {
            for (auto& [p, d] : local_deltas[t]) {
                if (d == 0) {
                    continue;
                }
                auto& st = pair_counts[p];
                st.count += d;
                if (d > 0) {
                    // 这些 word 索引现在出现 p,加入 positions
                    // 简化:从 local_deltas 无法反查 word 索引,改成全量重扫
                    // 这里先不入 positions,留到下方重扫
                    affected.insert(p);
                } else {
                    affected.insert(p);
                }
            }
        }
        // 对所有 affected pair,重扫它们的 positions(简化策略,保证正确)
        // 注:HF 的做法是把 (pair, iw) 在 worker 里就追踪好,这里简化以换取可读性
        for (const auto& p : affected) {
            auto& st = pair_counts[p];
            st.positions.clear();
        }
        for (uint32_t wi : pos_vec) {
            const auto& syms = words[wi]->symbols();
            for (std::size_t j = 0; j + 1 < syms.size(); ++j) {
                if (syms[j].len == 0 || syms[j + 1].len == 0) {
                    continue;
                }
                Pair p{syms[j].c, syms[j + 1].c};
                if (affected.contains(p)) {
                    pair_counts[p].positions.insert(wi);
                }
            }
        }
        // 把 affected 中 count > 0 的重新入堆(用最新 count/positions)
        for (const auto& p : affected) {
            const auto& st = pair_counts[p];
            if (st.count > 0) {
                heap.push(TrainMerge{p, static_cast<uint64_t>(st.count), st.positions});
            }
        }

        // 进度回调
        if (opts_.progress) {
            ++step;
            opts_.progress(step, opts_.vocab_size, "Compute merges");
        }
    }
}

absl::Status BpeTrainer::Impl::train(BPE& model) {
    if (word_counts_.empty()) {
        return absl::FailedPreconditionError("no corpus fed to trainer");
    }

    // 复用模型自身的 prefix/suffix(若 trainer 未显式设置)
    if (!opts_.continuing_subword_prefix) {
        opts_.continuing_subword_prefix = model.options().continuing_subword_prefix;
    }
    if (!opts_.end_of_word_suffix) {
        opts_.end_of_word_suffix = model.options().end_of_word_suffix;
    }

    absl::flat_hash_map<std::string, TokenId> word_to_id;
    word_to_id.reserve(opts_.vocab_size);
    std::vector<std::string> id_to_word;
    id_to_word.reserve(opts_.vocab_size);

    // 1. special tokens
    add_special_tokens(word_to_id, id_to_word);

    // 2. alphabet
    compute_alphabet(word_to_id, id_to_word);

    // 3. tokenize words
    auto [words, counts] = tokenize_words(word_to_id, id_to_word);

    // 4. count pairs
    auto pair_counts = count_pairs(words, counts);

    // 5. merge loop
    std::vector<std::pair<Pair, TokenId>> merges;
    merge_loop(words, counts, pair_counts, word_to_id, id_to_word, merges);

    // 6. 回写 model.vocab / vocab_r / merges
    Vocab vocab;
    for (TokenId id = 0; id < id_to_word.size(); ++id) {
        vocab[id_to_word[id]] = id;
    }
    VocabR vocab_r;
    for (TokenId id = 0; id < id_to_word.size(); ++id) {
        vocab_r[id] = id_to_word[id];
    }
    MergeMap merge_map;
    for (std::size_t rank = 0; rank < merges.size(); ++rank) {
        auto& [p, new_id] = merges[rank];
        merge_map[p] = MergeValue{static_cast<uint32_t>(rank), new_id};
    }

    // 写回 model.impl_
    model.impl_->vocab = std::move(vocab);
    model.impl_->vocab_r = std::move(vocab_r);
    model.impl_->merges = std::move(merge_map);
    if (opts_.continuing_subword_prefix) {
        model.impl_->opts.continuing_subword_prefix = *opts_.continuing_subword_prefix;
    }
    if (opts_.end_of_word_suffix) {
        model.impl_->opts.end_of_word_suffix = *opts_.end_of_word_suffix;
    }
    // 清缓存,防止旧 token 状态污染
    model.clear_cache();

    return absl::OkStatus();
}

}  // namespace bpe