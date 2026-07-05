// bpe/types.h — 公共基础类型
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace bpe {

using TokenId = uint32_t;

// token 字符串 → id(正向词表)
using Vocab = absl::flat_hash_map<std::string, TokenId>;
// id → token 字符串(反向,由 vocab build 时构建)
using VocabR = absl::flat_hash_map<TokenId, std::string>;
// 原始合并列表(加载 merges.txt / 训练输出用),顺序即 rank
using MergeList = std::vector<std::pair<std::string, std::string>>;

// 一对相邻 token id,是 MergeMap 的键
struct Pair {
    TokenId a;
    TokenId b;

    bool operator==(const Pair& o) const noexcept {
        return a == o.a && b == o.b;
    }

    bool operator!=(const Pair& o) const noexcept {
        return !(*this == o);
    }

    bool operator<(const Pair& o) const noexcept {
        return a != o.a ? a < o.a : b < o.b;
    }
};

// Pair 的 std::hash 兼容版本;也用于 absl 哈希容器(AbslHashValue 见下)
struct PairHash {
    std::size_t operator()(const Pair& p) const noexcept {
        // 折叠到 64 位,与设计文档一致
        return (static_cast<std::size_t>(p.a) << 32) | static_cast<std::size_t>(p.b);
    }
};

// absl::flat_hash_map<Pair, ...> 哈希支持
template <typename H>
H AbslHashValue(H h, const Pair& p) {
    return H::combine(std::move(h), p.a, p.b);
}

// 单个 token:id + 字面值 + 在原始字节流中的偏移区间
struct Token {
    TokenId id;
    std::string value;
    std::pair<uint32_t, uint32_t> offsets;  // [start, end),按字节计
};

// 一次编码的全部输出
struct Encoding {
    std::vector<TokenId> ids;
    std::vector<std::string> tokens;
    std::vector<std::pair<uint32_t, uint32_t>> offsets;
    std::vector<uint32_t> type_ids;
    std::vector<uint8_t> attention_mask;
    std::vector<uint8_t> special_tokens_mask;
};

}  // namespace bpe
