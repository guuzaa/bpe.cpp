// tests/test_word.cpp — M1: Word / Symbol / merge_all / merge 单测
#include <algorithm>
#include <gtest/gtest.h>

#include "models/bpe/word.h"
#include "utils/dheap.h"
#include "utils/unicode.h"

using namespace bpe;
using namespace bpe::bpe_internal;
using namespace bpe::util;

// ---- helpers ---------------------------------------------------------------
static Word make_word(std::vector<TokenId> ids, std::vector<uint32_t> lens) {
    assert(ids.size() == lens.size());
    Word w;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        w.add(ids[i], lens[i]);
    }
    return w;
}

static std::unordered_map<Pair, MergeValue, PairHash> make_merges(
    std::initializer_list<std::tuple<TokenId, TokenId, uint32_t, TokenId>> ms) {
    std::unordered_map<Pair, MergeValue, PairHash> out;
    for (auto [a, b, rank, mid] : ms) {
        out[Pair{a, b}] = MergeValue{rank, mid};
    }
    return out;
}

// ---- 基础构造 ---------------------------------------------------------------
TEST(Word, AddAndInspect) {
    auto w = make_word({1, 2, 3}, {1, 1, 1});
    EXPECT_EQ(w.size(), 3u);
    EXPECT_EQ(w.ids(), (std::vector<TokenId>{1, 2, 3}));
    auto off = w.offsets();
    ASSERT_EQ(off.size(), 3u);
    EXPECT_EQ(off[0], (std::pair<uint32_t, uint32_t>{0, 1}));
    EXPECT_EQ(off[1], (std::pair<uint32_t, uint32_t>{1, 2}));
    EXPECT_EQ(off[2], (std::pair<uint32_t, uint32_t>{2, 3}));
}

TEST(Word, Empty) {
    Word w;
    EXPECT_TRUE(w.empty());
    EXPECT_EQ(w.size(), 0u);
}

// ---- merge_all:单次合并 ----------------------------------------------------
TEST(Word, MergeAllSingle) {
    auto w = make_word({1, 2, 3}, {1, 1, 1});
    auto merges = make_merges({{1, 2, 0, 12}});
    w.merge_all(merges);
    EXPECT_EQ(w.ids(), (std::vector<TokenId>{12, 3}));
    EXPECT_EQ(w.offsets().size(), 2u);
    EXPECT_EQ(w.offsets()[0], (std::pair<uint32_t, uint32_t>{0, 2}));
    EXPECT_EQ(w.offsets()[1], (std::pair<uint32_t, uint32_t>{2, 3}));
}

// ---- merge_all:rank 决定优先级 ---------------------------------------------
TEST(Word, MergeAllRankPriority) {
    // 1 2 3;两个可能合并:(1,2) rank=1, (2,3) rank=0
    // 应先合并 (2,3) → 1 23;然后 (1,23) 不在 merges 中,终止
    auto w = make_word({1, 2, 3}, {1, 1, 1});
    auto merges = make_merges({{1, 2, 1, 12}, {2, 3, 0, 23}});
    w.merge_all(merges);
    EXPECT_EQ(w.ids(), (std::vector<TokenId>{1, 23}));
}

// ---- merge_all:链式合并 ----------------------------------------------------
TEST(Word, MergeAllChain) {
    // 1 2 3 4 → (1,2)→12,再 (12,3)→123,再 (123,4)→1234
    auto w = make_word({1, 2, 3, 4}, {1, 1, 1, 1});
    auto merges = make_merges({
        {1, 2, 0, 12},
        {12, 3, 1, 123},
        {123, 4, 2, 1234},
    });
    w.merge_all(merges);
    ASSERT_EQ(w.ids(), (std::vector<TokenId>{1234}));
    EXPECT_EQ(w.offsets()[0], (std::pair<uint32_t, uint32_t>{0, 4}));
}

// ---- merge_all:贪心(同 rank 多处可合并) ----------------------------------
TEST(Word, MergeAllGreedySameRank) {
    // 1 2 1 2 → (1,2) rank=0
    // 合并后:12 12,但 (12,12) 不在 merges 中,终止
    auto w = make_word({1, 2, 1, 2}, {1, 1, 1, 1});
    auto merges = make_merges({{1, 2, 0, 12}});
    w.merge_all(merges);
    EXPECT_EQ(w.ids(), (std::vector<TokenId>{12, 12}));
    EXPECT_EQ(w.offsets()[0], (std::pair<uint32_t, uint32_t>{0, 2}));
    EXPECT_EQ(w.offsets()[1], (std::pair<uint32_t, uint32_t>{2, 4}));
}

// ---- merge_all:惰性失效(堆中有过期条目) -----------------------------------
TEST(Word, MergeAllStaleInvalidation) {
    // 1 2 2 3,merges 中有 (1,2) 和 (2,2) 和 (2,3)
    // rank: (2,2)=0 优先 → 1 22 3;此时堆里若有 (2,3) 候选(基于初始 2 3),
    // 应被惰性校验丢弃
    auto w = make_word({1, 2, 2, 3}, {1, 1, 1, 1});
    auto merges = make_merges({
        {1, 2, 1, 12},
        {2, 2, 0, 22},
        {2, 3, 2, 23},
    });
    w.merge_all(merges);
    EXPECT_EQ(w.ids(), (std::vector<TokenId>{1, 22, 3}));
}

// ---- merge_all:dropout 应跳过部分合并 --------------------------------------
TEST(Word, MergeAllDropout) {
    // dropout=1.0 应跳过所有合并,保持原样
    auto w = make_word({1, 2, 3}, {1, 1, 1});
    auto merges = make_merges({{1, 2, 0, 12}, {2, 3, 1, 23}});
    w.merge_all(merges, 1.0f);
    EXPECT_EQ(w.ids(), (std::vector<TokenId>{1, 2, 3}));
}

TEST(Word, MergeAllNoDropout) {
    // dropout=0.0 应等价于无 dropout
    auto w = make_word({1, 2, 3}, {1, 1, 1});
    auto merges = make_merges({{1, 2, 0, 12}, {2, 3, 1, 23}});
    w.merge_all(merges, 0.0f);
    EXPECT_EQ(w.ids(), (std::vector<TokenId>{12, 3}));
}

// ---- merge(训练用):count delta 正确性 -------------------------------------
TEST(Word, TrainMergeReturnsDeltas) {
    // 1 2 3,合并 (2,3)→23
    // 邻居 delta:
    //   (1,2) 消失 -1,(1,23) 出现 +1
    //   右邻居:(3,?) 无,故无右 delta
    auto w = make_word({1, 2, 3}, {1, 1, 1});
    auto deltas = w.merge(2, 3, 23);
    EXPECT_EQ(w.ids(), (std::vector<TokenId>{1, 23}));

    // 验证 delta 集合
    std::unordered_map<Pair, int32_t, PairHash> dm;
    for (auto& [p, d] : deltas) {
        dm[p] += d;
    }
    EXPECT_EQ(dm[(Pair{1, 2})], -1);
    EXPECT_EQ(dm[(Pair{1, 23})], +1);
}

TEST(Word, TrainMergeMultipleOccurrences) {
    // 1 2 1 2,合并 (1,2)→12 应替换两处
    auto w = make_word({1, 2, 1, 2}, {1, 1, 1, 1});
    auto deltas = w.merge(1, 2, 12);
    EXPECT_EQ(w.ids(), (std::vector<TokenId>{12, 12}));
    // 两个 12 之间只有一对邻居关系,故 (12,12) 计数 +1;
    // (12,1) 在第一处合并时 +1(右邻居),在第二处合并时 -1(左邻居),净 0
    std::unordered_map<Pair, int32_t, PairHash> dm;
    for (auto& [p, d] : deltas) {
        dm[p] += d;
    }
    EXPECT_EQ(dm[(Pair{12, 12})], +1);
    EXPECT_EQ(dm[(Pair{12, 1})], 0);  // +1 与 -1 抵消,key 仍在但值为 0
}

TEST(Word, TrainMergeMaxTokenLength) {
    // 1 2 3,合并 (1,2)→12 长度 2 OK;但若 max_token_length=1 则跳过
    {
        auto w = make_word({1, 2, 3}, {1, 1, 1});
        auto deltas = w.merge(1, 2, 12, 1);
        EXPECT_EQ(w.ids(), (std::vector<TokenId>{1, 2, 3}));
        EXPECT_TRUE(deltas.empty());
    }
    {
        auto w = make_word({1, 2, 3}, {1, 1, 1});
        auto deltas = w.merge(1, 2, 12, 2);
        EXPECT_EQ(w.ids(), (std::vector<TokenId>{12, 3}));
        EXPECT_FALSE(deltas.empty());
    }
}

// ---- d_heap ----------------------------------------------------------------
TEST(DHeap, OrderAndTop) {
    d_heap<int, 4> h;  // 最大堆
    for (int v : {3, 1, 4, 1, 5, 9, 2, 6}) {
        h.push(v);
    }
    std::vector<int> out;
    while (!h.empty()) {
        out.push_back(h.top());
        h.pop();
    }
    EXPECT_TRUE(std::is_sorted(out.cbegin(), out.cend(), std::greater<>()));
}

TEST(DHeap, Empty) {
    d_heap<int, 4> h;
    EXPECT_TRUE(h.empty());
    h.push(1);
    EXPECT_FALSE(h.empty());
    h.pop();
    EXPECT_TRUE(h.empty());
}

// ---- UTF-8 -----------------------------------------------------------------
TEST(Unicode, DecodeAscii) {
    const char s[] = "A";
    auto [cp, len] = decode_utf8(s, s + 1);
    EXPECT_EQ(cp, 0x41u);
    EXPECT_EQ(len, 1);
}

TEST(Unicode, DecodeMultibyte) {
    // "中" = U+4E2D → E4 B8 AD
    const char s[] = "\xE4\xB8\xAD";
    auto [cp, len] = decode_utf8(s, s + 3);
    EXPECT_EQ(cp, 0x4E2Du);
    EXPECT_EQ(len, 3);
}

TEST(Unicode, DecodeEmoji) {
    // "😀" = U+1F600 → F0 9F 98 80
    const char s[] = "\xF0\x9F\x98\x80";
    auto [cp, len] = decode_utf8(s, s + 4);
    EXPECT_EQ(cp, 0x1F600u);
    EXPECT_EQ(len, 4);
}

TEST(Unicode, IterateCodepoints) {
    std::string_view s = "A中";
    codepoint_iter it(s);
    std::vector<uint32_t> cps;
    int total_bytes = 0;
    for (auto v : it) {
        cps.push_back(v.cp);
        total_bytes += v.len;
    }
    EXPECT_EQ(cps, (std::vector<uint32_t>{0x41, 0x4E2D}));
    EXPECT_EQ(total_bytes, 4);  // 1 + 3
}
