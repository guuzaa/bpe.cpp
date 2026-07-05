// models/bpe/word.h — BPE 编码/训练时的单 pre-token 工作状态
//
// 设计参考 HuggingFace tokenizers 的 `models/bpe/word.rs`:
//  - Symbol 是双向链表节点,以 std::int32_t 索引表示 prev/next
//  - Word 是 Symbol 的 std::vector 底座
//  - 合并通过置 len=0 + 重链接实现惰性删除,末尾再 retain
//
// 提供两套 merge:
//  - merge_all:编码热路径,优先队列驱动,按 (rank,pos) 升序弹出,惰性失效
//  - merge    :训练用,线性扫描替换 (c1,c2)→replacement,返回邻居 count delta
//
// 注意:本头文件是 BPE 模型的内部头,不进 include/bpe/ 公共 API。
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "bpe/types.h"
#include "utils/dheap.h"

namespace bpe::bpe_internal {

struct Symbol {
    TokenId c;          // token id
    std::int32_t prev;  // -1 = 无前驱
    std::int32_t next;  // -1 = 无后继
    uint32_t len;       // 子串字节长度;0 表示已合并删除
};

// MergeMap 的值:(rank, merged_id)。rank 越小越先合并。
struct MergeValue {
    uint32_t rank;
    TokenId merged_id;
};

// 编码端的合并候选(最大堆行为,弹出 (rank,pos) 最小者)
struct EncodeMerge {
    uint32_t pos;  // 符号索引
    uint32_t rank;
    TokenId new_id;

    // 反向比较:rank 小者优先,pos 小者优先;std::less → 最大堆 top 是最大值,
    // 因此这里返回 "this 比 o 小" 当 this 的 (rank,pos) 更大
    bool operator<(const EncodeMerge& o) const noexcept {
        if (rank != o.rank) {
            return rank > o.rank;
        }
        return pos > o.pos;
    }
};

class Word {
   public:
    Word() = default;

    // 在末尾追加一个符号(由 BPE::merge_word 按 UTF-8 字符依次调用)
    void add(TokenId id, uint32_t byte_len);

    // 把最后一个活跃符号的字节长度扩展 byte_len(fuse_unk 用:把相邻 UNK 合并成单个符号)
    // 必须保证 word 非空且最后一个活跃符号 c == 传入 id(调用方负责)。
    void extend_last(uint32_t byte_len);

    // 训练阶段使用:线性扫描替换所有相邻 (c1,c2) 为 replacement。
    // 返回邻居 pair 的 count delta:[(pair, +1/-1)] 用于更新 pair_counts。
    // max_token_length 若设置,则跳过会形成超过此长度的合并。
    std::vector<std::pair<Pair, int32_t>> merge(TokenId c1, TokenId c2, TokenId replacement,
                                                std::optional<uint32_t> max_token_length = std::nullopt);

    // 编码阶段使用:优先队列驱动的合并。
    // dropout 若为 p>0,则每个候选以概率 p 被跳过(实现 BPE-Dropout)。
    // MergeMap 只需支持 find(begin/end),兼容 std::unordered_map 与 absl::flat_hash_map
    template <typename MergeMapT>
    void merge_all(const MergeMapT& merges, std::optional<float> dropout = std::nullopt);

    // 输出
    const std::vector<Symbol>& symbols() const noexcept {
        return symbols_;
    }

    std::size_t size() const noexcept {
        std::size_t n = 0;
        for (const auto& s : symbols_) {
            if (s.len != 0) {
                ++n;
            }
        }
        return n;
    }

    bool empty() const noexcept {
        return size() == 0;
    }

    // 返回当前所有活跃 token 的 id 序列(按顺序)
    std::vector<TokenId> ids() const;

    // 返回每个活跃符号的字节偏移累计(用于构造 Token.offsets)
    // 例:符号长度 [1,2,3] → 偏移 [(0,1),(1,3),(3,6)]
    std::vector<std::pair<uint32_t, uint32_t>> offsets() const;

   private:
    template <typename MergeMapT>
    bool try_push_pair(std::int32_t i, const MergeMapT& merges, util::d_heap<EncodeMerge, 4>& heap) const;

    void compact();  // 移除 len==0 的死符号(在 merge_all 末尾调用)

    // 安全下标:int32_t → size_t 的显式转换(调用方须保证 i >= 0)
    Symbol& sym(std::int32_t i) {
        return symbols_[static_cast<std::size_t>(i)];
    }

    const Symbol& sym(std::int32_t i) const {
        return symbols_[static_cast<std::size_t>(i)];
    }

    std::vector<Symbol> symbols_{};
};

// ---- 模板实现 ---------------------------------------------------------------
template <typename MergeMapT>
bool Word::try_push_pair(std::int32_t i, const MergeMapT& merges, util::d_heap<EncodeMerge, 4>& heap) const {
    if (i < 0) {
        return false;
    }
    if (sym(i).len == 0) {
        return false;
    }
    std::int32_t j = sym(i).next;
    if (j < 0 || sym(j).len == 0) {
        return false;
    }
    auto it = merges.find(Pair{sym(i).c, sym(j).c});
    if (it == merges.end()) {
        return false;
    }
    heap.push(EncodeMerge{
        .pos = static_cast<uint32_t>(i),
        .rank = it->second.rank,
        .new_id = it->second.merged_id,
    });
    return true;
}

template <typename MergeMapT>
void Word::merge_all(const MergeMapT& merges, std::optional<float> dropout) {
    if (symbols_.empty()) {
        return;
    }

    // 线程局部 RNG(dropout 用);不强制可重现性
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    util::d_heap<EncodeMerge, 4> heap;
    std::vector<EncodeMerge> skipped;
    skipped.reserve(8);

    // 初始:扫描所有相邻对(链表遍历,i >= 0 表示还有符号)
    for (std::int32_t i = 0; i >= 0; i = sym(i).next) {
        if (sym(i).len == 0) {
            continue;
        }
        try_push_pair(i, merges, heap);
    }

    while (!heap.empty()) {
        EncodeMerge m = heap.top();
        heap.pop();

        // 惰性校验 1:pos 处符号已死
        if (static_cast<std::size_t>(m.pos) >= symbols_.size() || sym(static_cast<std::int32_t>(m.pos)).len == 0) {
            continue;
        }
        std::int32_t i = static_cast<std::int32_t>(m.pos);
        std::int32_t j = sym(i).next;
        // 惰性校验 2:j 已死或越界
        if (j < 0 || sym(j).len == 0) {
            continue;
        }
        // 惰性校验 3:当前 (i,j) 的 pair 在 merges 中仍是 m.new_id
        auto it = merges.find(Pair{sym(i).c, sym(j).c});
        if (it == merges.end() || it->second.merged_id != m.new_id) {
            continue;
        }

        // BPE-Dropout:按概率跳过
        if (dropout && *dropout > 0.0f && dist(rng) < *dropout) {
            skipped.push_back(m);
            continue;
        }

        // 执行合并
        sym(i).c = m.new_id;
        sym(i).len += sym(j).len;
        sym(i).next = sym(j).next;
        sym(j).len = 0;
        std::int32_t n2 = sym(i).next;
        if (n2 >= 0) {
            sym(n2).prev = i;
        }

        // 推入新生成的邻居对
        try_push_pair(sym(i).prev, merges, heap);
        try_push_pair(i, merges, heap);

        // 回灌被 dropout 跳过的候选
        if (!skipped.empty()) {
            for (const auto& s : skipped) {
                heap.push(s);
            }
            skipped.clear();
        }
    }

    compact();
}

}  // namespace bpe::bpe_internal
