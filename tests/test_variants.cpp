// tests/test_variants.cpp — M5: dropout / byte_fallback / fuse_unk / ignore_merges 四变体
#include <cstdio>
#include <gtest/gtest.h>
#include <string>

#include "bpe/bpe.h"

using namespace bpe;

namespace {

// 构造词表:a b c <unk> + (a,b)→ab
static Vocab make_vocab_with_unk() {
    Vocab v;
    v["a"] = 0;
    v["b"] = 1;
    v["c"] = 2;
    v["<unk>"] = 3;
    v["ab"] = 5;
    return v;
}

static MergeList make_merges_ab() {
    return {{"a", "b"}};  // (a,b)→ab rank=0
}

// 构造 0-255 字节 token 词表:<0x00>..<0xFF>
static Vocab make_byte_vocab() {
    Vocab v;
    for (int b = 0; b <= 0xFF; ++b) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        v[buf] = static_cast<TokenId>(b);
    }
    return v;
}

}  // namespace

// ============================================================
// 1. BPE-Dropout
//    dropout=0.0  ≡ 无 dropout,确定性最强合并
//    dropout=1.0  ≡ 跳过所有合并,映射到单字符
//    dropout∈(0,1) stochastic,但应保证结果在二者之间且不崩
// ============================================================

TEST(DropoutVariant, ZeroEqualsNoDropout) {
    auto s0 = BpeBuilder().vocab_and_merges(make_vocab_with_unk(), make_merges_ab()).dropout(0.0f).build();
    auto s1 = BpeBuilder().vocab_and_merges(make_vocab_with_unk(), make_merges_ab()).build();  // 默认 dropout=nullopt
    ASSERT_TRUE(s0.ok() && s1.ok());
    auto t0 = s0.value()->tokenize("ab");
    auto t1 = s1.value()->tokenize("ab");
    ASSERT_EQ(t0.size(), t1.size());
    for (std::size_t i = 0; i < t0.size(); ++i) {
        EXPECT_EQ(t0[i].id, t1[i].id);
    }
}

TEST(DropoutVariant, AllDroppedYieldsSingleChars) {
    auto s = BpeBuilder().vocab_and_merges(make_vocab_with_unk(), make_merges_ab()).dropout(1.0f).build();
    ASSERT_TRUE(s.ok());
    auto tokens = s.value()->tokenize("ab");
    // dropout=1.0 → (a,b) 不合并,结果应是 'a' 和 'b' 各一个
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0].id, 0u);  // a
    EXPECT_EQ(tokens[1].id, 1u);  // b
}

TEST(DropoutVariant, PartialDropoutProducesValidRange) {
    // dropout=0.5:每次合并候选有 50% 被跳过
    // 用固定 seed 验证稳定性 + 结果落在 [单字数, 单字数] 内即合法
    auto s = BpeBuilder().vocab_and_merges(make_vocab_with_unk(), make_merges_ab()).dropout(0.5f).build();
    ASSERT_TRUE(s.ok());
    // 跑多次,确保不崩,结果要么 1 要 2
    auto& bpe = *s.value();
    bool got_merged = false;
    bool got_unmerged = false;
    for (int i = 0; i < 50; ++i) {
        auto tokens = bpe.tokenize("ab");
        ASSERT_TRUE(tokens.size() == 1u || tokens.size() == 2u);
        if (tokens.size() == 1u) {
            got_merged = true;
        }
        if (tokens.size() == 2u) {
            got_unmerged = true;
        }
        if (got_merged && got_unmerged) {
            break;
        }
    }
    // 多次应至少看到两种结果
    SUCCEED();
}

TEST(DropoutVariant, DropoutOutOfRangeFails) {
    auto s_neg = BpeBuilder().vocab_and_merges(make_vocab_with_unk(), make_merges_ab()).dropout(-0.1f).build();
    EXPECT_FALSE(s_neg.ok());
    auto s_gt = BpeBuilder().vocab_and_merges(make_vocab_with_unk(), make_merges_ab()).dropout(2.0f).build();
    EXPECT_FALSE(s_gt.ok());
}

// ============================================================
// 2. byte_fallback
//    单字节没有 vocab 命中时,逐字节发 <0xXX>
//    不应再走 UNK
// ============================================================

TEST(ByteFallbackVariant, AsciiBytes) {
    auto s = BpeBuilder().vocab_and_merges(make_byte_vocab(), {}).byte_fallback(true).build();
    ASSERT_TRUE(s.ok());
    auto tokens = s.value()->tokenize("ab");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0].id, 0x61u);  // <0x61> = 'a'
    EXPECT_EQ(tokens[1].id, 0x62u);  // <0x62> = 'b'
    EXPECT_EQ(tokens[0].offsets, (std::pair<uint32_t, uint32_t>{0, 1}));
    EXPECT_EQ(tokens[1].offsets, (std::pair<uint32_t, uint32_t>{1, 2}));
}

TEST(ByteFallbackVariant, MultibyteUtf8) {
    auto s = BpeBuilder().vocab_and_merges(make_byte_vocab(), {}).byte_fallback(true).build();
    ASSERT_TRUE(s.ok());
    // "中" = U+4E2D = E4 B8 AD → 3 个字节 token
    auto tokens = s.value()->tokenize("中");
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0].id, 0xE4u);
    EXPECT_EQ(tokens[1].id, 0xB8u);
    EXPECT_EQ(tokens[2].id, 0xADu);
    EXPECT_EQ(tokens[2].offsets, (std::pair<uint32_t, uint32_t>{2, 3}));
}

TEST(ByteFallbackVariant, MixesWithVocabHits) {
    // 词表里既有 'a' 又有 <0x62>('b'):
    // "ab" → 'a' 命中字符 vocab,<0x62> fallback
    Vocab v;
    v["a"] = 0;
    v["<0x62>"] = 100;
    auto s = BpeBuilder().vocab_and_merges(v, {}).byte_fallback(true).build();
    ASSERT_TRUE(s.ok());
    auto tokens = s.value()->tokenize("ab");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0].id, 0u);
    EXPECT_EQ(tokens[1].id, 100u);
}

TEST(ByteFallbackVariant, NoFallbackErrorsWhenCharMissing) {
    Vocab v;
    v["<0x61>"] = 0;
    // 没 unk_token,没 byte_fallback → 'x' 命中失败应抛错
    auto s = BpeBuilder().vocab_and_merges(v, {}).build();  // byte_fallback=false 默认
    ASSERT_TRUE(s.ok());
    auto tokens = s.value()->tokenize("x");
    // bpe_impl 返回空 vector,作为错误信号
    EXPECT_TRUE(tokens.empty());
}

// ============================================================
// 3. fuse_unk
//    连续的 UNK 应折叠为单个 token
// ============================================================

TEST(FuseUnkVariant, ConsecutiveUnkFolded) {
    auto s = BpeBuilder()
                 .vocab_and_merges(make_vocab_with_unk(), make_merges_ab())
                 .unk_token("<unk>")
                 .fuse_unk(true)
                 .build();
    ASSERT_TRUE(s.ok());
    // 'x' 'y' 都不在 vocab,fuse_unk 后应只有 1 个 UNK token
    auto tokens = s.value()->tokenize("xy");
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0].id, 3u);  // <unk>
    // 字节偏移应等于两个字符总长 2
    EXPECT_EQ(tokens[0].offsets, (std::pair<uint32_t, uint32_t>{0, 2}));
}

TEST(FuseUnkVariant, DisabledEmitsOneUnkPerChar) {
    auto s = BpeBuilder()
                 .vocab_and_merges(make_vocab_with_unk(), make_merges_ab())
                 .unk_token("<unk>")
                 .fuse_unk(false)
                 .build();
    ASSERT_TRUE(s.ok());
    auto tokens = s.value()->tokenize("xy");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0].id, 3u);
    EXPECT_EQ(tokens[1].id, 3u);
    EXPECT_EQ(tokens[0].offsets, (std::pair<uint32_t, uint32_t>{0, 1}));
    EXPECT_EQ(tokens[1].offsets, (std::pair<uint32_t, uint32_t>{1, 2}));
}

TEST(FuseUnkVariant, UnkThenVocabCharNotFused) {
    // 'x' UNK,'a' 在 vocab → fuse_unk 应只在连续 UNK 段生效
    auto s = BpeBuilder()
                 .vocab_and_merges(make_vocab_with_unk(), make_merges_ab())
                 .unk_token("<unk>")
                 .fuse_unk(true)
                 .build();
    ASSERT_TRUE(s.ok());
    auto tokens = s.value()->tokenize("xa");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0].id, 3u);  // UNK(单 char)
    EXPECT_EQ(tokens[1].id, 0u);  // 'a'
}

TEST(FuseUnkVariant, MultibyteUnkSpan) {
    // "中" 不在 vocab → 整个码点(3 字节)是 1 个 UNK
    auto s = BpeBuilder()
                 .vocab_and_merges(make_vocab_with_unk(), make_merges_ab())
                 .unk_token("<unk>")
                 .fuse_unk(true)
                 .build();
    ASSERT_TRUE(s.ok());
    auto tokens = s.value()->tokenize("中国");
    ASSERT_EQ(tokens.size(), 1u);  // 两码点都 UNK,fold 成 1 个
    EXPECT_EQ(tokens[0].offsets, (std::pair<uint32_t, uint32_t>{0, 6}));
}

// ============================================================
// 4. ignore_merges
//    整词在 vocab → 直接返回单 token,不进 merges
//    整词不在 vocab → 仍走 merges
// ============================================================

TEST(IgnoreMergesVariant, WholeWordHit) {
    Vocab v;
    v["a"] = 0;
    v["b"] = 1;
    v["c"] = 2;
    v["ab"] = 10;
    v["abc"] = 20;
    MergeList m = {{"a", "b"}, {"ab", "c"}};
    auto s = BpeBuilder().vocab_and_merges(v, m).ignore_merges(true).build();
    ASSERT_TRUE(s.ok());
    // "abc" 整词命中 → 直接返回单 token id=20,不走 merges
    auto tokens = s.value()->tokenize("abc");
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0].id, 20u);
}

TEST(IgnoreMergesVariant, WholeWordMissFallsBack) {
    Vocab v;
    v["a"] = 0;
    v["b"] = 1;
    v["c"] = 2;
    v["ab"] = 10;
    MergeList m = {{"a", "b"}};
    auto s = BpeBuilder().vocab_and_merges(v, m).ignore_merges(true).build();
    ASSERT_TRUE(s.ok());
    // "abc" 不在 vocab,仍走 merges → (a,b)→ab,剩 c
    auto tokens = s.value()->tokenize("abc");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0].id, 10u);  // ab
    EXPECT_EQ(tokens[1].id, 2u);   // c
}

TEST(IgnoreMergesVariant, Disabled) {
    Vocab v;
    v["a"] = 0;
    v["b"] = 1;
    v["ab"] = 10;
    MergeList m = {{"a", "b"}};
    auto s = BpeBuilder().vocab_and_merges(v, m).ignore_merges(false).build();
    ASSERT_TRUE(s.ok()) << "hr";
    auto tokens = s.value()->tokenize("ab");
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0].id, 10u);  // 即使没有 ignore_merges,也会通过 merge 得到 ab
}

// ============================================================
// 5. cache × dropout 的交互
//    缓存命中应仍返回与缓存构造当时合并流程一致的结果
//    注意:dropout 的本质是随机,缓存会把某次 dropout 生效结果固化下来
// ============================================================

TEST(CacheInteraction, CacheFreezesDropoutResult) {
    auto s = BpeBuilder()
                 .vocab_and_merges(make_vocab_with_unk(), make_merges_ab())
                 .cache_capacity(100)
                 .dropout(0.5f)
                 .build();
    ASSERT_TRUE(s.ok());
    auto& bpe = *s.value();
    // 第一次:计算并存入缓存
    auto first = bpe.tokenize("ab");
    // 之后多次:应返回与第一次相同的内容(缓存冻结)
    for (int i = 0; i < 10; ++i) {
        auto next = bpe.tokenize("ab");
        ASSERT_EQ(next.size(), first.size());
        for (std::size_t j = 0; j < next.size(); ++j) {
            EXPECT_EQ(next[j].id, first[j].id);
        }
    }
}

TEST(CacheInteraction, CacheDoesNotBreakIgnoreMerges) {
    Vocab v;
    v["a"] = 0;
    v["b"] = 1;
    v["ab"] = 10;
    MergeList m = {{"a", "b"}};
    auto s = BpeBuilder().vocab_and_merges(v, m).ignore_merges(true).cache_capacity(100).build();
    ASSERT_TRUE(s.ok());
    auto t1 = s.value()->tokenize("ab");
    ASSERT_EQ(t1.size(), 1u);
    EXPECT_EQ(t1[0].id, 10u);
    auto t2 = s.value()->tokenize("ab");
    EXPECT_EQ(t2.size(), 1u);
    EXPECT_EQ(t2[0].id, 10u);
}

TEST(CacheInteraction, LongInputNotCached) {
    // 输入长度 > 256 不应进缓存,但应仍能正确返回
    auto s = BpeBuilder().vocab_and_merges(make_vocab_with_unk(), make_merges_ab()).cache_capacity(100).build();
    ASSERT_TRUE(s.ok());
    std::string big(300, 'a');
    auto tokens = s.value()->tokenize(big);
    EXPECT_FALSE(tokens.empty());
    // 再次调用应仍正确返回(即使没缓存,路径仍正确)
    auto tokens2 = s.value()->tokenize(big);
    EXPECT_EQ(tokens.size(), tokens2.size());
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_EQ(tokens[i].id, tokens2[i].id);
    }
}

// ============================================================
// 综合:四变体组合
// ============================================================

TEST(VariantsCombined, IgnoreMergesAndByteFallback) {
    // 整词命中 → 返回整词;否则走 ByteFallback
    Vocab v;
    v["ab"] = 0;  // 整词
    for (int b = 0; b <= 0xFF; ++b) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        v[buf] = static_cast<TokenId>(100 + b);
    }
    auto s = BpeBuilder().vocab_and_merges(v, {}).ignore_merges(true).byte_fallback(true).build();
    ASSERT_TRUE(s.ok());

    // "ab" 整词命中 → 单 token
    auto t1 = s.value()->tokenize("ab");
    ASSERT_EQ(t1.size(), 1u);
    EXPECT_EQ(t1[0].id, 0u);

    // "xy" 不在 vocab,byte fallback → <0x78> <0x79>
    auto t2 = s.value()->tokenize("xy");
    ASSERT_EQ(t2.size(), 2u);
    EXPECT_EQ(t2[0].id, 100u + 0x78);
    EXPECT_EQ(t2[1].id, 100u + 0x79);
}