// models/bpe/word.cpp — Word 实现
#include "models/bpe/word.h"
#include <limits>

namespace bpe::bpe_internal {

void Word::add(TokenId id, uint32_t byte_len) {
    assert(symbols_.size() < static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()));
    const auto idx = static_cast<std::int32_t>(symbols_.size());
    if (!symbols_.empty()) {
        symbols_.back().next = idx;
    }
    symbols_.push_back(Symbol{id, idx - 1, -1, byte_len});
    ++live_count_;
}

void Word::extend_last(uint32_t byte_len) {
    // 从末尾向前找到第一个活跃符号
    for (std::int32_t i = static_cast<std::int32_t>(symbols_.size()) - 1; i >= 0; --i) {
        if (sym(i).len != 0) {
            sym(i).len += byte_len;
            return;
        }
    }
    // 无活跃符号:回退为 add
    add(0, byte_len);
}

std::vector<std::pair<Pair, int32_t>> Word::merge(TokenId c1, TokenId c2, TokenId replacement,
                                                  std::optional<uint32_t> max_token_length) {
    std::vector<std::pair<Pair, int32_t>> changes;
    if (symbols_.empty()) {
        return changes;
    }

    // 链表顺序扫描:i >= 0 表示还有活跃符号;跳过 len==0 的死符号
    for (std::int32_t i = 0; i >= 0;) {
        if (sym(i).len == 0) {
            i = sym(i).next;
            continue;
        }
        std::int32_t j = sym(i).next;
        // j 可能是死符号(理论上合并后 i.next 已跳过死的 j,这里保险起见跳过)
        while (j >= 0 && sym(j).len == 0) {
            j = sym(j).next;
        }
        if (j < 0) {
            break;  // i 是末位活跃符号,无右邻居
        }
        if (sym(i).c == c1 && sym(j).c == c2) {
            const uint32_t new_len = sym(i).len + sym(j).len;
            if (max_token_length && new_len > *max_token_length) {
                i = j;  // 跳过此对,继续扫描
                continue;
            }
            // 左邻居对 (prev, c1) 消失,(prev, replacement) 出现
            std::int32_t p = sym(i).prev;
            if (p >= 0 && sym(p).len != 0) {
                changes.emplace_back(Pair{sym(p).c, c1}, -1);
                changes.emplace_back(Pair{sym(p).c, replacement}, +1);
            }
            // 右邻居对 (c2, next2) 消失,(replacement, next2) 出现
            std::int32_t n2 = sym(j).next;
            while (n2 >= 0 && sym(n2).len == 0) {
                n2 = sym(n2).next;
            }
            if (n2 >= 0) {
                changes.emplace_back(Pair{c2, sym(n2).c}, -1);
                changes.emplace_back(Pair{replacement, sym(n2).c}, +1);
            }
            // 执行合并:i 吸收 j
            sym(i).c = replacement;
            sym(i).len = new_len;
            sym(i).next = n2;
            sym(j).len = 0;  // 标记死符号
            --live_count_;
            if (n2 >= 0) {
                sym(n2).prev = i;
            }
            // 不前进 i,可能新对 (i, n2) 也可合并
            continue;
        }
        i = j;
    }
    return changes;
}

void Word::compact() {
    // 剔除死符号,直接写成稠密双向链
    std::vector<Symbol> kept;
    kept.reserve(live_count_);
    for (const auto& s : symbols_) {
        if (s.len != 0) {
            kept.push_back(s);
        }
    }
    const std::int32_t n = static_cast<std::int32_t>(kept.size());
    for (std::int32_t i = 0; i < n; ++i) {
        kept[static_cast<std::size_t>(i)].prev = i - 1;
        kept[static_cast<std::size_t>(i)].next = (i + 1 < n) ? i + 1 : -1;
    }
    symbols_ = std::move(kept);
    live_count_ = symbols_.size();
}

std::vector<TokenId> Word::ids() const {
    std::vector<TokenId> out;
    out.reserve(live_count_);
    for (const auto& s : symbols_) {
        if (s.len != 0) {
            out.push_back(s.c);
        }
    }
    return out;
}

std::vector<std::pair<uint32_t, uint32_t>> Word::offsets() const {
    std::vector<std::pair<uint32_t, uint32_t>> out;
    out.reserve(live_count_);
    uint32_t cur = 0;
    for (const auto& s : symbols_) {
        if (s.len == 0) {
            continue;
        }
        out.emplace_back(cur, cur + s.len);
        cur += s.len;
    }
    return out;
}

}  // namespace bpe::bpe_internal