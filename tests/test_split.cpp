// tests/test_split.cpp — Split & Sequence PreTokenizer 单测
//
// 黄金用例对照 HuggingFace tokenizers:
//   1) HF `tokenizers/src/tokenizer/normalizer.rs` 中对 split 行为的注释:
//      "the-final--countdown" 切分示例
//   2) HF `tokenizers/src/pre_tokenizers/split.rs` 的 `basic` 测试:
//      pattern = r"\w+|[^\w\s]+", invert=true 在 "How are you doing?" 上的各行为
//   3) DeepSeek v3.2 pretokenizer.json 中的 Sequence = [Split, Split, Split, ByteLevel]
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "bpe/interfaces.h"
#include "bpe/pretokenizers.h"
#include "pretokenizers/byte_level.h"
#include "pretokenizers/sequence.h"
#include "pretokenizers/split.h"

using namespace bpe;
using namespace bpe::pretokenizers;

// 工具:把 PreToken 列表简化为 (text, offsets) 对,便于断言
static std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>> to_pairs(const std::vector<PreToken>& toks) {
    std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>> out;
    out.reserve(toks.size());
    for (const auto& t : toks) {
        out.emplace_back(t.text, t.offsets);
    }
    return out;
}

// ---- HF "the-final--countdown" '-' Isolated 各行为 ---------------------------
// 对照 HF normalizer.rs 注释:pattern='-', invert=false(默认)
static const char* kDash = "-";
static const std::string kTheFinal = "the-final--countdown";

// positions:  t(0..3) -(3..4) f(4..9) -(9..10) -(10..11) c(11..20)

TEST(Split, DashRemoved) {
    Split sp(kDash, SplitPatternKind::kString, SplitBehavior::kRemoved, false);
    PreTokenizedString input({PreToken{kTheFinal, {0, 20}, {}}});
    sp.pre_tokenize(input);
    auto out = to_pairs(input.tokens());
    // 与 HF 一致:["the", "final", "countdown"]
    EXPECT_EQ(out, (std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>>{
                       {"the", {0, 3}},
                       {"final", {4, 9}},
                       {"countdown", {11, 20}},
                   }));
}

TEST(Split, DashIsolated) {
    Split sp(kDash, SplitPatternKind::kString, SplitBehavior::kIsolated, false);
    PreTokenizedString input({PreToken{kTheFinal, {0, 20}, {}}});
    sp.pre_tokenize(input);
    auto out = to_pairs(input.tokens());
    EXPECT_EQ(out, (std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>>{
                       {"the", {0, 3}},
                       {"-", {3, 4}},
                       {"final", {4, 9}},
                       {"-", {9, 10}},
                       {"-", {10, 11}},
                       {"countdown", {11, 20}},
                   }));
}

TEST(Split, DashMergedWithPrevious) {
    Split sp(kDash, SplitPatternKind::kString, SplitBehavior::kMergedWithPrevious, false);
    PreTokenizedString input({PreToken{kTheFinal, {0, 20}, {}}});
    sp.pre_tokenize(input);
    auto out = to_pairs(input.tokens());
    EXPECT_EQ(out, (std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>>{
                       {"the-", {0, 4}},
                       {"final-", {4, 10}},
                       {"-", {10, 11}},
                       {"countdown", {11, 20}},
                   }));
}

TEST(Split, DashMergedWithNext) {
    Split sp(kDash, SplitPatternKind::kString, SplitBehavior::kMergedWithNext, false);
    PreTokenizedString input({PreToken{kTheFinal, {0, 20}, {}}});
    sp.pre_tokenize(input);
    auto out = to_pairs(input.tokens());
    EXPECT_EQ(out, (std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>>{
                       {"the", {0, 3}},
                       {"-final", {3, 9}},
                       {"-", {9, 10}},
                       {"-countdown", {10, 20}},
                   }));
}

TEST(Split, DashContiguous) {
    Split sp(kDash, SplitPatternKind::kString, SplitBehavior::kContiguous, false);
    PreTokenizedString input({PreToken{kTheFinal, {0, 20}, {}}});
    sp.pre_tokenize(input);
    auto out = to_pairs(input.tokens());
    EXPECT_EQ(out, (std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>>{
                       {"the", {0, 3}},
                       {"-", {3, 4}},
                       {"final", {4, 9}},
                       {"--", {9, 11}},
                       {"countdown", {11, 20}},
                   }));
}

// ---- HF split.rs `basic` 测试:regex `\w+|[^\w\s]+`, invert=true ------------
// 期望:["How", "are", "you", "doing", "?"]
TEST(Split, RegexInvertedIsolated) {
    Split sp(R"(\w+|[^\w\s]+)", SplitPatternKind::kRegex, SplitBehavior::kIsolated, true);
    PreTokenizedString input({PreToken{"How are you doing?", {0, 18}, {}}});
    sp.pre_tokenize(input);
    auto out = to_pairs(input.tokens());
    EXPECT_EQ(out, (std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>>{
                       {"How", {0, 3}},
                       {" ", {3, 4}},
                       {"are", {4, 7}},
                       {" ", {7, 8}},
                       {"you", {8, 11}},
                       {" ", {11, 12}},
                       {"doing", {12, 17}},
                       {"?", {17, 18}},
                   }));
}

TEST(Split, RegexInvertedRemoved) {
    Split sp(R"(\w+|[^\w\s]+)", SplitPatternKind::kRegex, SplitBehavior::kRemoved, true);
    PreTokenizedString input({PreToken{"How are you doing?", {0, 18}, {}}});
    sp.pre_tokenize(input);
    auto out = to_pairs(input.tokens());
    EXPECT_EQ(out, (std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>>{
                       {"How", {0, 3}},
                       {"are", {4, 7}},
                       {"you", {8, 11}},
                       {"doing", {12, 17}},
                       {"?", {17, 18}},
                   }));
}

TEST(Split, RegexInvertedMergedWithPrevious) {
    Split sp(R"(\w+|[^\w\s]+)", SplitPatternKind::kRegex, SplitBehavior::kMergedWithPrevious, true);
    PreTokenizedString input({PreToken{"How are you doing?", {0, 18}, {}}});
    sp.pre_tokenize(input);
    auto out = to_pairs(input.tokens());
    EXPECT_EQ(out, (std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>>{
                       {"How ", {0, 4}},
                       {"are ", {4, 8}},
                       {"you ", {8, 12}},
                       {"doing", {12, 17}},
                       {"?", {17, 18}},
                   }));
}

TEST(Split, RegexInvertedMergedWithNext) {
    Split sp(R"(\w+|[^\w\s]+)", SplitPatternKind::kRegex, SplitBehavior::kMergedWithNext, true);
    PreTokenizedString input({PreToken{"How are you doing?", {0, 18}, {}}});
    sp.pre_tokenize(input);
    auto out = to_pairs(input.tokens());
    EXPECT_EQ(out, (std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>>{
                       {"How", {0, 3}},
                       {" are", {3, 7}},
                       {" you", {7, 11}},
                       {" doing", {11, 17}},
                       {"?", {17, 18}},
                   }));
}

TEST(Split, RegexInvertedContiguous) {
    Split sp(R"(\w+|[^\w\s]+)", SplitPatternKind::kRegex, SplitBehavior::kContiguous, true);
    PreTokenizedString input({PreToken{"How are you doing?", {0, 18}, {}}});
    sp.pre_tokenize(input);
    auto out = to_pairs(input.tokens());
    EXPECT_EQ(out, (std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>>{
                       {"How", {0, 3}},
                       {" ", {3, 4}},
                       {"are", {4, 7}},
                       {" ", {7, 8}},
                       {"you", {8, 11}},
                       {" ", {11, 12}},
                       {"doing?", {12, 18}},
                   }));
}

// ---- invert 替换:Split("Hello") invert=true ≡ Split(" ") invert=false ----
// 来源于 HF split.rs `invert` 测试
TEST(Split, InvertEquivalence) {
    Split span(" ", SplitPatternKind::kString, SplitBehavior::kRemoved, false);
    Split span_invert("Hello", SplitPatternKind::kString, SplitBehavior::kRemoved, true);
    PreTokenizedString input1({PreToken{"Hello Hello Hello", {0, 17}, {}}});
    PreTokenizedString input2({PreToken{"Hello Hello Hello", {0, 17}, {}}});
    span.pre_tokenize(input1);
    span_invert.pre_tokenize(input2);
    EXPECT_EQ(to_pairs(input1.tokens()), to_pairs(input2.tokens()));
}

// ---- 工厂出口:make_split_pretokenizer--------------------------------------
TEST(SplitFactory, ConstructsViaBpeEnums) {
    auto pt = make_split_pretokenizer(R"(\s+)", SplitPatternKind::kRegex, SplitBehavior::kRemoved, false);
    ASSERT_NE(pt, nullptr);
    PreTokenizedString input({PreToken{"a b c", {0, 5}, {}}});
    pt->pre_tokenize(input);
    auto out = to_pairs(input.tokens());
    EXPECT_EQ(out, (std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>>{
                       {"a", {0, 1}},
                       {"b", {2, 3}},
                       {"c", {4, 5}},
                   }));
}

// ---- Sequence 顺序组合 -----------------------------------------------------
TEST(Sequence, AppliesEachInOrder) {
    std::vector<std::unique_ptr<PreTokenizer>> subs;
    // Split on digits 1-3, Isolated, invert=false → 把数字段隔出
    subs.push_back(std::make_unique<Split>(R"(\p{N}{1,3})", SplitPatternKind::kRegex, SplitBehavior::kIsolated, false));
    // Split on whitespace, Removed, invert=false → 再去掉空格
    subs.push_back(std::make_unique<Split>(R"(\s+)", SplitPatternKind::kRegex, SplitBehavior::kRemoved, false));
    Sequence seq(std::move(subs));
    // "abc123 de456" → Split(\p{N}{1,3}, Isolated): ["abc","123"," ","de","456"]
    //                 → Split(\s+, Removed):         ["abc","123",   "de","456"]
    PreTokenizedString input({PreToken{"abc123 de456", {0, 11}, {}}});
    seq.pre_tokenize(input);
    auto out = to_pairs(input.tokens());
    EXPECT_EQ(out, (std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>>{
                       {"abc", {0, 3}},
                       {"123", {3, 6}},
                       {"de", {7, 9}},
                       {"456", {9, 12}},
                   }));
}

// ---- DeepSeek v3.2 完整 pretokenizer 复现 ----------------------------------
// 复刻 DeepSeek v3.2 tokenizer.json 的:
//   pretokenizers = [
//     Split(pattern=Regex "\\p{N}{1,3}", behavior=Isolated, invert=false),
//     Split(pattern=Regex "[一-龥぀-ゟ゠-ヿ]+", behavior=Isolated, invert=false),
//     Split(pattern=Regex "<main regex>", behavior=Isolated, invert=false),
//     ByteLevel(add_prefix_space=false, trim_offsets=true, use_regex=false),
//   ]
TEST(DeepSeekPipeline, SplitsThenByteLevel) {
    const std::string G = "\xC4\xA0";  // "Ġ" 字节字符
    const char* main_regex =
        R"([!"#$%&'()*+,\-./:;<=>?@\[\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

    std::vector<std::unique_ptr<PreTokenizer>> subs;
    subs.push_back(std::make_unique<Split>(R"(\p{N}{1,3})", SplitPatternKind::kRegex, SplitBehavior::kIsolated, false));
    subs.push_back(
        std::make_unique<Split>("[一-龥぀-ゟ゠-ヿ]+", SplitPatternKind::kRegex, SplitBehavior::kIsolated, false));
    subs.push_back(std::make_unique<Split>(main_regex, SplitPatternKind::kRegex, SplitBehavior::kIsolated, false));
    subs.push_back(std::make_unique<ByteLevel>(false, true, false));
    Sequence seq(std::move(subs));

    // 测试样本:"hello 12345 世界! abc"
    // 期望流程(简化分析):
    //   1) Split digits Isolated : "hello " + "123" + "45" + " 世界! abc"(非数字段若不为空也成段)
    //   2) Split CJK+  Isolated  : 在上一步基础上,把连续 CJK 段独立成 token
    //   3) Split main regex      : 按 HF 主正则把单词/标点解耦
    //   4) ByteLevel use_regex=false: 每段原字节映射为 byte-char(空格→Ġ)
    //
    // 这里主要验证两件事:
    //   a. 数字 12345 被切成 "123" + "45"(连续运行最多 3 位)
    //   b. CJK "世界" 独立成 PreToken,经 ByteLevel 映射后仍是 UTF-8 字节("世界")
    //   c. 空格被 byte-char 化为 Ġ;标点 ! 独立成段
    PreTokenizedString input({PreToken{"hello 12345 世界! abc", {0, 23}, {}}});
    seq.pre_tokenize(input);
    const auto& toks = input.tokens();

    // 收集 PreToken 的 text(注意 ByteLevel 之后空格已是 Ġ,中文已映射为 byte-char)
    std::vector<std::string> texts;
    texts.reserve(toks.size());
    for (const auto& t : toks) {
        texts.push_back(t.text);
    }

    // 数字应被切成 123 / 45(ASCII,ByteLevel 后保持原样)
    ASSERT_TRUE(std::find(texts.begin(), texts.end(), "123") != texts.end());
    ASSERT_TRUE(std::find(texts.begin(), texts.end(), "45") != texts.end());
    // 中文段应独立成 PreToken,经 ByteLevel 后等同 ByteLevel::encode_bytes("世界")
    const std::string world_byte = ByteLevel::encode_bytes("世界");
    ASSERT_FALSE(world_byte.empty());
    ASSERT_TRUE(std::find(texts.begin(), texts.end(), world_byte) != texts.end());
    // 标点 '!' 独立成 PreToken
    ASSERT_TRUE(std::find(texts.begin(), texts.end(), "!") != texts.end());
    // 至少存在一个带 Ġ 前缀的英文段(空格映射后的 "Ġabc")
    ASSERT_TRUE(std::find(texts.begin(), texts.end(), G + "abc") != texts.end());

    // 所有 PreToken 的 alignment 此时已非空(ByteLevel 填充),长度 = text.size()+字节匹配
    // 这里只校验 offsets 严格递增且非退化
    for (std::size_t i = 0; i + 1 < toks.size(); ++i) {
        EXPECT_LE(toks[i].offsets.first, toks[i].offsets.second);
        EXPECT_LE(toks[i].offsets.second, toks[i + 1].offsets.first) << "tokens 不应交叠/逆序 @ i=" << i;
    }
    if (!toks.empty()) {
        EXPECT_EQ(toks.front().offsets.first, 0u);
        EXPECT_EQ(toks.back().offsets.second, 23u);
    }
}