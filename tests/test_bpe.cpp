// tests/test_bpe.cpp — M2: BPE 端到端测试(构造微型 vocab + merges,验证 encode)
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include "bpe/bpe.h"

using namespace bpe;

// ---- helpers ---------------------------------------------------------------
// 构造一个微型词表:
//   单字符:a b c d → id 0-3
//   合并:(a,b)→"ab" id=4, (ab,c)→"abc" id=5, (c,d)→"cd" id=6
static Vocab make_vocab() {
    Vocab v;
    v["a"] = 0;
    v["b"] = 1;
    v["c"] = 2;
    v["d"] = 3;
    v["ab"] = 4;
    v["abc"] = 5;
    v["cd"] = 6;
    return v;
}

static MergeList make_merges() {
    return {
        {"a", "b"},   // rank 0 → ab(4)
        {"ab", "c"},  // rank 1 → abc(5)
        {"c", "d"},   // rank 2 → cd(6)
    };
}

// ---- 基础构造 ---------------------------------------------------------------
TEST(BpeBuilder, BuildFromVocabAndMerges) {
    auto status = BpeBuilder().vocab_and_merges(make_vocab(), make_merges()).build();
    ASSERT_TRUE(status.ok()) << status.status().message();
    auto bpe = std::move(*status);
    EXPECT_EQ(bpe->vocab_size(), 7u);
    EXPECT_EQ(bpe->token_to_id("a"), 0u);
    EXPECT_EQ(bpe->token_to_id("abc"), 5u);
    EXPECT_EQ(bpe->id_to_token(5), std::optional<std::string>("abc"));
    EXPECT_FALSE(bpe->token_to_id("xyz").has_value());
}

TEST(BpeBuilder, BuildWithoutVocabFails) {
    auto status = BpeBuilder().build();
    EXPECT_FALSE(status.ok());
}

TEST(BpeBuilder, BuildWithEmptyMergesOk) {
    Vocab v;
    v["x"] = 0;
    v["y"] = 1;
    auto status = BpeBuilder().vocab_and_merges(v, {}).build();
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(status.value()->vocab_size(), 2u);
}

TEST(BpeBuilder, DropoutOutOfRangeFails) {
    auto status = BpeBuilder().vocab_and_merges(make_vocab(), make_merges()).dropout(1.5f).build();
    EXPECT_FALSE(status.ok());
}

// ---- encode 端到端 ---------------------------------------------------------
TEST(BpeEncode, SimpleChain) {
    // "abc" → a b c → (a,b) rank=0 合并 → ab c → (ab,c) rank=1 合并 → abc
    auto status = BpeBuilder().vocab_and_merges(make_vocab(), make_merges()).build();
    ASSERT_TRUE(status.ok());
    auto& bpe = *status.value();

    auto tokens = bpe.tokenize("abc");
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0].id, 5u);  // "abc"
    EXPECT_EQ(tokens[0].value, "abc");
    EXPECT_EQ(tokens[0].offsets, (std::pair<uint32_t, uint32_t>{0, 3}));
}

TEST(BpeEncode, TwoGroups) {
    // "abcd" → a b c d → (a,b) rank=0 → ab c d → (c,d) rank=2 → ab cd
    //         → (ab,c) rank=1 → abc d → 无更多合并
    // 注意:rank 1 < rank 2,所以 (ab,c) 先于 (c,d) 应用
    // 但初始时 (c,d) 的 rank=2 比 (ab,c) 的 rank=1 大,先弹 (ab,c)
    // 结果:abc d
    auto status = BpeBuilder().vocab_and_merges(make_vocab(), make_merges()).build();
    ASSERT_TRUE(status.ok());
    auto& bpe = *status.value();

    auto tokens = bpe.tokenize("abcd");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0].id, 5u);  // "abc"
    EXPECT_EQ(tokens[1].id, 3u);  // "d"
    EXPECT_EQ(tokens[0].offsets, (std::pair<uint32_t, uint32_t>{0, 3}));
    EXPECT_EQ(tokens[1].offsets, (std::pair<uint32_t, uint32_t>{3, 4}));
}

TEST(BpeEncode, NoMergePossible) {
    // "dcba" → d c b a,无任何相邻对在 merges 中
    auto status = BpeBuilder().vocab_and_merges(make_vocab(), make_merges()).build();
    ASSERT_TRUE(status.ok());
    auto& bpe = *status.value();

    auto tokens = bpe.tokenize("dcba");
    ASSERT_EQ(tokens.size(), 4u);
    EXPECT_EQ(tokens[0].id, 3u);  // d
    EXPECT_EQ(tokens[1].id, 2u);  // c
    EXPECT_EQ(tokens[2].id, 1u);  // b
    EXPECT_EQ(tokens[3].id, 0u);  // a
}

TEST(BpeEncode, Empty) {
    auto status = BpeBuilder().vocab_and_merges(make_vocab(), make_merges()).build();
    ASSERT_TRUE(status.ok());
    auto tokens = status.value()->tokenize("");
    EXPECT_TRUE(tokens.empty());
}

// ---- ignore_merges ---------------------------------------------------------
TEST(BpeEncode, IgnoreMergesFastPath) {
    // ignore_merges=true:"ab" 整词在 vocab 中,直接返回单 token,不走 merges
    auto status = BpeBuilder().vocab_and_merges(make_vocab(), make_merges()).ignore_merges(true).build();
    ASSERT_TRUE(status.ok());
    auto& bpe = *status.value();

    auto tokens = bpe.tokenize("ab");
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0].id, 4u);  // "ab" 整词
    EXPECT_EQ(tokens[0].value, "ab");
}

TEST(BpeEncode, IgnoreMergesFallthrough) {
    // ignore_merges=true 但整词不在 vocab:"abcd" 仍走 merges
    auto status = BpeBuilder().vocab_and_merges(make_vocab(), make_merges()).ignore_merges(true).build();
    ASSERT_TRUE(status.ok());
    auto& bpe = *status.value();

    auto tokens = bpe.tokenize("abcd");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0].id, 5u);  // abc
}

// ---- cache -----------------------------------------------------------------
TEST(BpeEncode, CacheHitReturnsSameResult) {
    auto status = BpeBuilder().vocab_and_merges(make_vocab(), make_merges()).cache_capacity(100).build();
    ASSERT_TRUE(status.ok());
    auto& bpe = *status.value();

    auto t1 = bpe.tokenize("abc");
    auto t2 = bpe.tokenize("abc");
    ASSERT_EQ(t1.size(), t2.size());
    for (std::size_t i = 0; i < t1.size(); ++i) {
        EXPECT_EQ(t1[i].id, t2[i].id);
        EXPECT_EQ(t1[i].value, t2[i].value);
        EXPECT_EQ(t1[i].offsets, t2[i].offsets);
    }
}

TEST(BpeEncode, CacheDisabledWorks) {
    auto status = BpeBuilder().vocab_and_merges(make_vocab(), make_merges()).cache_capacity(0).build();
    ASSERT_TRUE(status.ok());
    auto tokens = status.value()->tokenize("abc");
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0].id, 5u);
}

TEST(BpeEncode, ClearCachePreservesCorrectness) {
    auto status = BpeBuilder().vocab_and_merges(make_vocab(), make_merges()).cache_capacity(100).build();
    ASSERT_TRUE(status.ok());
    auto& bpe = *status.value();

    (void)bpe.tokenize("abc");
    bpe.clear_cache();
    auto tokens = bpe.tokenize("abc");
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0].id, 5u);
}

// ---- UTF-8 多字节字符 -------------------------------------------------------
TEST(BpeEncode, MultibyteChar) {
    // 词表含 "中" 和 "国" 以及合并 "中国"
    Vocab v;
    v["中"] = 0;
    v["国"] = 1;
    v["中国"] = 2;
    MergeList m = {{"中", "国"}};

    auto status = BpeBuilder().vocab_and_merges(v, m).build();
    ASSERT_TRUE(status.ok());
    auto& bpe = *status.value();

    auto tokens = bpe.tokenize("中国");
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0].id, 2u);
    EXPECT_EQ(tokens[0].value, "中国");
    // "中" 3 字节,"国" 3 字节
    EXPECT_EQ(tokens[0].offsets, (std::pair<uint32_t, uint32_t>{0, 6}));
}

// ---- end_of_word_suffix ----------------------------------------------------
TEST(BpeEncode, EndOfWordSuffix) {
    // Sennrich 风格:</w> 后缀
    // 词表:h(0) e(1) l(2) o</w>(3) he(4) lo</w>(5) hello</w>(6)
    Vocab v;
    v["h"] = 0;
    v["e"] = 1;
    v["l"] = 2;
    v["o</w>"] = 3;
    v["he"] = 4;
    v["lo</w>"] = 5;
    v["hello</w>"] = 6;
    MergeList m = {
        {"h", "e"},         // he
        {"l", "o</w>"},     // lo</w>
        {"he", "l"},        // hel
        {"hel", "lo</w>"},  // hello</w> — 但 hel 不在 vocab,需补
    };
    // 补充 hel
    v["hel"] = 7;
    auto status = BpeBuilder().vocab_and_merges(v, m).end_of_word_suffix("</w>").build();
    ASSERT_TRUE(status.ok());
    auto& bpe = *status.value();

    // "hello" → h e l l o</w> → merges 后应得到 hello</w>
    auto tokens = bpe.tokenize("hello");
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0].id, 6u);
    EXPECT_EQ(tokens[0].value, "hello</w>");
    EXPECT_EQ(tokens[0].offsets, (std::pair<uint32_t, uint32_t>{0, 5}));
}

// ---- byte_fallback ---------------------------------------------------------
TEST(BpeEncode, ByteFallback) {
    // 词表含 <0x61> (= 'a') 但不含 "a" 本身
    Vocab v;
    v["<0x61>"] = 0;  // 'a'
    v["<0x62>"] = 1;  // 'b'
    auto status = BpeBuilder().vocab_and_merges(v, {}).byte_fallback(true).build();
    ASSERT_TRUE(status.ok());
    auto& bpe = *status.value();

    auto tokens = bpe.tokenize("ab");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0].id, 0u);  // <0x61>
    EXPECT_EQ(tokens[1].id, 1u);  // <0x62>
    EXPECT_EQ(tokens[0].offsets, (std::pair<uint32_t, uint32_t>{0, 1}));
    EXPECT_EQ(tokens[1].offsets, (std::pair<uint32_t, uint32_t>{1, 2}));
}

// ---- unk_token -------------------------------------------------------------
TEST(BpeEncode, UnkToken) {
    Vocab v;
    v["a"] = 0;
    v["<unk>"] = 1;
    auto status = BpeBuilder().vocab_and_merges(v, {}).unk_token("<unk>").build();
    ASSERT_TRUE(status.ok());
    auto& bpe = *status.value();

    // "a" 在 vocab 中
    auto tokens = bpe.tokenize("a");
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0].id, 0u);

    // "x" 不在 vocab,且无 byte_fallback,应发 UNK
    auto tokens2 = bpe.tokenize("x");
    ASSERT_EQ(tokens2.size(), 1u);
    EXPECT_EQ(tokens2[0].id, 1u);
    EXPECT_EQ(tokens2[0].value, "<unk>");
}

// ---- 文件加载 ---------------------------------------------------------------

class BpeFileTest : public ::testing::Test {
   protected:
    std::string tmpdir;

    void SetUp() override {
        tmpdir = std::filesystem::temp_directory_path().string() + "/bpe_m2_" +
                 std::to_string(reinterpret_cast<uintptr_t>(this)) + "/";
        std::filesystem::create_directories(tmpdir);
    }

    void TearDown() override {
        std::filesystem::remove_all(tmpdir);
    }

    void write_file(const std::string& name, const std::string& content) {
        std::ofstream(tmpdir + name) << content;
    }
};

TEST_F(BpeFileTest, LoadVocabJsonAndMergesTxt) {
    write_file("vocab.json", R"({"a":0,"b":1,"ab":2})");
    write_file("merges.txt", "a b\n");

    auto status = BpeBuilder().files(tmpdir + "vocab.json", tmpdir + "merges.txt").build();
    ASSERT_TRUE(status.ok()) << status.status().message();
    auto& bpe = *status.value();

    EXPECT_EQ(bpe.vocab_size(), 3u);
    auto tokens = bpe.tokenize("ab");
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0].id, 2u);
}

TEST_F(BpeFileTest, MissingVocabJsonFails) {
    auto status = BpeBuilder().files(tmpdir + "nope.json", tmpdir + "nope.txt").build();
    EXPECT_FALSE(status.ok());
}

TEST_F(BpeFileTest, MergesWithCommentLine) {
    write_file("vocab.json", R"({"a":0,"b":1,"ab":2})");
    write_file("merges.txt", "#version: 0.1\na b\n");

    auto status = BpeBuilder().files(tmpdir + "vocab.json", tmpdir + "merges.txt").build();
    ASSERT_TRUE(status.ok());
    auto tokens = status.value()->tokenize("ab");
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0].id, 2u);
}
