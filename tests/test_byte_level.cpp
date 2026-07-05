// tests/test_byte_level.cpp — M3: ByteLevel PreTokenizer/Decoder/PostProcessor
//
// 黄金用例对照 HF tokenizers:
//   3rdparty/tokenizers/tokenizers/src/pre_tokenizers/byte_level.rs:245+
#include <gtest/gtest.h>

#include "bpe/decoders.h"
#include "bpe/interfaces.h"
#include "bpe/pretokenizers.h"
#include "pretokenizers/byte_level.h"
#include "utils/bytes_char.h"

using namespace bpe;
using namespace bpe::pretokenizers;
using namespace bpe::util;

// ---- bytes_char 映射 -------------------------------------------------------
TEST(BytesChar, SpaceMapsToU0120) {
    // 空格 0x20 → U+0120 'Ġ'(GPT-2 标志性字符)
    EXPECT_EQ(bytes_to_chars()[0x20], 0x120u);
    // 反向
    EXPECT_EQ(char_to_byte(0x120), 0x20);
}

TEST(BytesChar, PrintableAscii) {
    // 'a' = 0x61 → 自身
    EXPECT_EQ(bytes_to_chars()[0x61], 0x61u);
    EXPECT_EQ(char_to_byte(0x61), 0x61);
}

TEST(BytesChar, ControlByteMappedToHighArea) {
    // 0x00 → U+0100(第一个映射到高位的控制字节)
    EXPECT_EQ(bytes_to_chars()[0x00], 0x100u);
    EXPECT_EQ(char_to_byte(0x100), 0x00);
    // 0x01 → U+0101
    EXPECT_EQ(bytes_to_chars()[0x01], 0x101u);
    EXPECT_EQ(char_to_byte(0x101), 0x01);
}

TEST(BytesChar, UnmappedCodepointReturnsNeg) {
    // U+0041 已映射,但 U+5000 不在映射表中
    EXPECT_EQ(char_to_byte(0x5000), -1);
}

TEST(BytesChar, RoundTripAllBytes) {
    // 全部 256 个字节双向映射闭合
    const auto& fwd = bytes_to_chars();
    for (int b = 0; b <= 0xFF; ++b) {
        uint32_t cp = fwd[static_cast<uint8_t>(b)];
        EXPECT_EQ(char_to_byte(cp), b) << "byte 0x" << std::hex << b << " → cp U+" << cp << " → byte 不一致";
    }
}

TEST(BytesChar, ByteToStringAscii) {
    EXPECT_EQ(byte_to_string('a'), "a");
    EXPECT_EQ(byte_to_string('!'), "!");
}

TEST(BytesChar, ByteToStringSpace) {
    // 空格 → U+0120 = 0xC4 0xA0
    EXPECT_EQ(byte_to_string(' '), std::string("\xC4\xA0"));
}

// ---- ByteLevel PreTokenizer:HF 黄金用例 -----------------------------------
// 对照 HF test pre_tokenization:add_prefix_space=false
TEST(ByteLevelPreTokenize, HfGoldenNoPrefixSpace) {
    ByteLevel bl(/*add_prefix_space=*/false, /*trim_offsets=*/false,
                 /*use_regex=*/true);
    PreTokenizedString input({PreToken{"Hello my friend, how is your day going?", {0, 39}, {}}});
    bl.pre_tokenize(input);

    const auto& toks = input.tokens();
    ASSERT_EQ(toks.size(), 10u);
    // 期望:Hello / Ġmy / Ġfriend / , / Ġhow / Ġis / Ġyour / Ġday / Ġgoing / ?
    // byte-char 表示:空格 = Ġ (U+0120, UTF-8: 0xC4 0xA0)
    const std::string G = "\xC4\xA0";  // "Ġ"
    EXPECT_EQ(toks[0].text, "Hello");
    EXPECT_EQ(toks[0].offsets, (std::pair<uint32_t, uint32_t>{0, 5}));
    EXPECT_EQ(toks[1].text, G + "my");
    EXPECT_EQ(toks[1].offsets, (std::pair<uint32_t, uint32_t>{5, 8}));
    EXPECT_EQ(toks[2].text, G + "friend");
    EXPECT_EQ(toks[2].offsets, (std::pair<uint32_t, uint32_t>{8, 15}));
    EXPECT_EQ(toks[3].text, ",");
    EXPECT_EQ(toks[3].offsets, (std::pair<uint32_t, uint32_t>{15, 16}));
    EXPECT_EQ(toks[4].text, G + "how");
    EXPECT_EQ(toks[4].offsets, (std::pair<uint32_t, uint32_t>{16, 20}));
    EXPECT_EQ(toks[5].text, G + "is");
    EXPECT_EQ(toks[5].offsets, (std::pair<uint32_t, uint32_t>{20, 23}));
    EXPECT_EQ(toks[6].text, G + "your");
    EXPECT_EQ(toks[6].offsets, (std::pair<uint32_t, uint32_t>{23, 28}));
    EXPECT_EQ(toks[7].text, G + "day");
    EXPECT_EQ(toks[7].offsets, (std::pair<uint32_t, uint32_t>{28, 32}));
    EXPECT_EQ(toks[8].text, G + "going");
    EXPECT_EQ(toks[8].offsets, (std::pair<uint32_t, uint32_t>{32, 38}));
    EXPECT_EQ(toks[9].text, "?");
    EXPECT_EQ(toks[9].offsets, (std::pair<uint32_t, uint32_t>{38, 39}));
}

// 对照 HF test pre_tokenization_no_regex
TEST(ByteLevelPreTokenize, HfGoldenNoRegex) {
    ByteLevel bl(/*add_prefix_space=*/true, /*trim_offsets=*/false,
                 /*use_regex=*/false);
    PreTokenizedString input({PreToken{"Hello my friend, how is your day going?", {0, 39}, {}}});
    bl.pre_tokenize(input);

    const auto& toks = input.tokens();
    ASSERT_EQ(toks.size(), 1u);
    // add_prefix_space=true → 前加空格 → 整体映射
    // " Hello..." 共 40 字节,空格 → Ġ
    const std::string G = "\xC4\xA0";
    EXPECT_EQ(toks[0].text,
              G + "Hello" + G + "my" + G + "friend," + G + "how" + G + "is" + G + "your" + G + "day" + G + "going?");
    // 偏移:整段(因 add_prefix_space 加了 1 字节,offsets 仍指向原文 [0,39])
    EXPECT_EQ(toks[0].offsets, (std::pair<uint32_t, uint32_t>{0, 39}));
}

// 对照 HF test add_prefix_space:add_prefix_space=true
TEST(ByteLevelPreTokenize, HfGoldenWithPrefixSpace) {
    ByteLevel bl(/*add_prefix_space=*/true, /*trim_offsets=*/false,
                 /*use_regex=*/true);
    // 输入无前导空格
    PreTokenizedString input({PreToken{"Hello my friend, how is your day going?", {0, 39}, {}}});
    bl.pre_tokenize(input);

    const auto& toks = input.tokens();
    ASSERT_EQ(toks.size(), 10u);
    const std::string G = "\xC4\xA0";
    // 第一个 token "Hello" 前被加了空格 → "ĠHello"
    EXPECT_EQ(toks[0].text, G + "Hello");
    // 偏移:由于加了 prefix space,split 0 的原文偏移应仍指向 "Hello" 起点 0
    EXPECT_EQ(toks[0].offsets, (std::pair<uint32_t, uint32_t>{0, 5}));
    EXPECT_EQ(toks[1].text, G + "my");
    EXPECT_EQ(toks[1].offsets, (std::pair<uint32_t, uint32_t>{5, 8}));
}

// add_prefix_space=true 且输入已有前导空格(不应再加)
TEST(ByteLevelPreTokenize, PrefixSpaceAlreadyHasSpace) {
    ByteLevel bl(/*add_prefix_space=*/true, /*trim_offsets=*/false,
                 /*use_regex=*/true);
    PreTokenizedString input({PreToken{" Hello", {0, 6}, {}}});
    bl.pre_tokenize(input);

    const auto& toks = input.tokens();
    ASSERT_EQ(toks.size(), 1u);
    const std::string G = "\xC4\xA0";
    EXPECT_EQ(toks[0].text, G + "Hello");
    EXPECT_EQ(toks[0].offsets, (std::pair<uint32_t, uint32_t>{0, 6}));
}

// ---- ByteLevel Decoder:HF 黄金用例 ----------------------------------------
TEST(ByteLevelDecode, HfGolden) {
    ByteLevel bl(/*add_prefix_space=*/false, /*trim_offsets=*/false);
    const std::string G = "\xC4\xA0";
    std::vector<std::string> tokens = {"Hello",  G + "my",   G + "friend", ",",         G + "how",
                                       G + "is", G + "your", G + "day",    G + "going", "?"};
    std::string out = bl.decode(tokens);
    EXPECT_EQ(out, "Hello my friend, how is your day going?");
}

TEST(ByteLevelDecode, NonByteCharPreserved) {
    // 含未映射字符(如 '中')时,保留原字节
    ByteLevel bl(false, false);
    std::vector<std::string> tokens = {"中"};
    std::string out = bl.decode(tokens);
    EXPECT_EQ(out, "中");
}

TEST(ByteLevelDecode, EmptyTokens) {
    ByteLevel bl(false, false);
    std::vector<std::string> tokens;
    EXPECT_EQ(bl.decode(tokens), "");
}

// ---- ByteLevel PostProcessor:trim_offsets ---------------------------------
TEST(ByteLevelProcess, TrimOffsetsBasic) {
    // 构造一个 token,前导/后导各有一个空格(byte-char Ġ)
    ByteLevel bl(/*add_prefix_space=*/false, /*trim_offsets=*/true);
    const std::string G = "\xC4\xA0";
    Token t;
    t.value = G + "abc" + G;  // "ĠabcĠ"
    t.offsets = {10, 17};     // 假设原文对应 7 字节
    std::vector<Token> in = {t};
    auto out = bl.process(std::move(in));
    // 前导 1 空格 → offsets.first += 1;后导 1 空格 → offsets.second -= 1
    EXPECT_EQ(out[0].offsets, (std::pair<uint32_t, uint32_t>{11, 16}));
}

TEST(ByteLevelProcess, NoTrimWhenDisabled) {
    ByteLevel bl(/*add_prefix_space=*/false, /*trim_offsets=*/false);
    const std::string G = "\xC4\xA0";
    Token t;
    t.value = G + "abc" + G;
    t.offsets = {10, 17};
    std::vector<Token> in = {t};
    auto out = bl.process(std::move(in));
    EXPECT_EQ(out[0].offsets, (std::pair<uint32_t, uint32_t>{10, 17}));
}

TEST(ByteLevelProcess, FirstTokenPrefixSpacePreserved) {
    // add_prefix_space=true:首个 token 的 1 个前导空格应保留(我们加的)
    ByteLevel bl(/*add_prefix_space=*/true, /*trim_offsets=*/true);
    const std::string G = "\xC4\xA0";
    Token t;
    t.value = G + "Hello";  // 1 个前导空格
    t.offsets = {0, 5};     // 原文 "Hello"(prefix space 是我们加的)
    std::vector<Token> in = {t};
    auto out = bl.process(std::move(in));
    // 不应裁剪前导
    EXPECT_EQ(out[0].offsets.first, 0u);
}

TEST(ByteLevelProcess, MultipleLeadingSpacesTrimmed) {
    // add_prefix_space=true 但 leading=2 > 1 → 第一个是 "我们加的" 保留,另一个裁剪
    ByteLevel bl(/*add_prefix_space=*/true, /*trim_offsets=*/true);
    const std::string G = "\xC4\xA0";
    Token t;
    t.value = G + G + "Hello";  // 2 个前导空格
    t.offsets = {0, 6};         // 原文 " Hello"(6 字节)
    std::vector<Token> in = {t};
    auto out = bl.process(std::move(in));
    // leading=2,is_first && add_prefix_space && leading==1 不成立(leading==2)
    // → leading 保持 2 → first += 2
    EXPECT_EQ(out[0].offsets.first, 2u);
}

// ---- 工厂 ------------------------------------------------------------------
TEST(ByteLevelFactory, PreTokenizer) {
    auto pt = make_byte_level_pretokenizer(false, false, true);
    ASSERT_NE(pt, nullptr);
    PreTokenizedString input({PreToken{"Hello", {0, 5}, {}}});
    pt->pre_tokenize(input);
    ASSERT_EQ(input.tokens().size(), 1u);
    EXPECT_EQ(input.tokens()[0].text, "Hello");
}

TEST(ByteLevelFactory, Decoder) {
    auto dec = make_byte_level_decoder();
    ASSERT_NE(dec, nullptr);
    const std::string G = "\xC4\xA0";
    std::vector<std::string> tokens = {"Hello", G + "world"};
    EXPECT_EQ(dec->decode(tokens), "Hello world");
}

// ---- 端到端:ByteLevel PreTokenizer + BPE + ByteLevel Decoder round-trip ----
#include "bpe/bpe.h"

TEST(ByteLevelEndToEnd, EncodeDecodeRoundTrip) {
    // 构造一个 byte-level BPE:词表里用 byte-char 表示的字符
    // "Hello" → byte-char → 'H' 'e' 'l' 'l' 'o'(都是 ASCII,映射后仍是自身)
    // 加 merges 让 'l' 'l' → 'll'
    const std::string G = "\xC4\xA0";
    Vocab v;
    v["H"] = 0;
    v["e"] = 1;
    v["l"] = 2;
    v["o"] = 3;
    v["ll"] = 4;
    v[G + "Hello"] = 5;  // "ĠHello" 整词(byte-char 表示)
    v[G + "world"] = 6;  // "Ġworld" 整词
    MergeList m = {
        {"l", "l"},  // ll(rank 0)
    };
    auto bpe_status = BpeBuilder().vocab_and_merges(v, m).ignore_merges(true).build();
    ASSERT_TRUE(bpe_status.ok()) << bpe_status.status().message();
    auto& bpe = *bpe_status.value();

    // 1. pre-tokenize "Hello world"
    auto pt = make_byte_level_pretokenizer(/*add_prefix_space=*/true,
                                           /*trim_offsets=*/false,
                                           /*use_regex=*/true);
    PreTokenizedString input({PreToken{"Hello world", {0, 11}, {}}});
    pt->pre_tokenize(input);

    // 应得到两个 pre-token:"ĠHello" / "Ġworld"
    ASSERT_EQ(input.tokens().size(), 2u);
    EXPECT_EQ(input.tokens()[0].text, G + "Hello");
    EXPECT_EQ(input.tokens()[1].text, G + "world");

    // 2. BPE tokenize 每个 pre-token
    std::vector<std::string> token_strs;
    for (const auto& ptk : input.tokens()) {
        auto toks = bpe.tokenize(ptk.text);
        for (auto& t : toks) {
            token_strs.push_back(t.value);
        }
    }

    // 3. decode
    auto dec = make_byte_level_decoder();
    std::string recovered = dec->decode(token_strs);

    // 期望:" Hello world"(因 add_prefix_space=true)
    EXPECT_EQ(recovered, " Hello world");
}
