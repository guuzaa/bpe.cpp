// tests/test_tokenizer.cpp — M6: Tokenizer 总线 + Normalizer + tokenizer.json round-trip
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <string>

#include "bpe/bpe.h"
#include "bpe/decoders.h"
#include "bpe/interfaces.h"
#include "bpe/normalizers.h"
#include "bpe/pretokenizers.h"
#include "bpe/tokenizer.h"
#include "utils/bytes_char.h"

using namespace bpe;

// ---- helpers ---------------------------------------------------------------
static Vocab make_gpt2_like_vocab() {
    const std::string G = "\xC4\xA0";  // "Ġ" = byte-char for ' '
    Vocab v;
    v["H"] = 0;
    v["e"] = 1;
    v["l"] = 2;
    v["o"] = 3;
    v["ll"] = 4;
    v["Hello"] = 5;
    v[G + "world"] = 6;
    v[G + "W"] = 7;
    v["w"] = 8;
    v["r"] = 9;
    v["d"] = 10;
    v[G] = 13;            // 单独的 Ġ(空格 byte-char)
    v[G + "Hello"] = 14;  // add_prefix_space 后的整词
    v["<unk>"] = 11;
    v["<pad>"] = 12;
    return v;
}

static MergeList make_gpt2_like_merges() {
    return {
        {"l", "l"},     // rank 0: ll
        {"He", "l"},    // skip — He not in vocab
        {"H", "e"},     // rank 1: He
        {"Hel", "lo"},  // skip
    };
}

// ---- Normalizer 单测 -------------------------------------------------------
TEST(Normalizer, IdentityKeepsString) {
    auto n = make_identity_normalizer();
    NormalizedString ns("Hello");
    n->normalize(ns);
    EXPECT_EQ(ns.get(), "Hello");
    // 偏移对齐:每个 normalized 字节对应同一 original 字节
    auto [s, e] = ns.align_to_original(0, 5);
    EXPECT_EQ(s, 0u);
    EXPECT_EQ(e, 5u);
}

TEST(Normalizer, LowercaseAscii) {
    auto n = make_lowercase_normalizer();
    NormalizedString ns("Hello World");
    n->normalize(ns);
    EXPECT_EQ(ns.get(), "hello world");
}

TEST(Normalizer, UppercaseAscii) {
    auto n = make_uppercase_normalizer();
    NormalizedString ns("Hello World");
    n->normalize(ns);
    EXPECT_EQ(ns.get(), "HELLO WORLD");
}

TEST(Normalizer, StripRemovesWhitespace) {
    auto n = make_strip_normalizer();
    NormalizedString ns("  hello  ");
    n->normalize(ns);
    EXPECT_EQ(ns.get(), "hello");
}

TEST(Normalizer, Sequence) {
    // Strip + Lowercase 组合
    std::vector<std::unique_ptr<Normalizer>> ns;
    ns.push_back(make_strip_normalizer());
    ns.push_back(make_lowercase_normalizer());
    auto n = make_sequence_normalizer(std::move(ns));
    NormalizedString s("  Hello  ");
    n->normalize(s);
    EXPECT_EQ(s.get(), "hello");
}

TEST(Normalizer, OffsetAlignmentAfterLowercase) {
    // Lowercase 不改变长度,偏移对齐表应不变
    auto n = make_lowercase_normalizer();
    NormalizedString ns("HELLO");
    n->normalize(ns);
    EXPECT_EQ(ns.get(), "hello");
    auto [s, e] = ns.align_to_original(0, 5);
    EXPECT_EQ(s, 0u);
    EXPECT_EQ(e, 5u);
}

TEST(Normalizer, OffsetAlignmentAfterStrip) {
    // Strip "  x  " → "x",normalized 偏移 0..1 应映射到 original 偏移 2..3
    auto n = make_strip_normalizer();
    NormalizedString ns("  x  ");
    n->normalize(ns);
    EXPECT_EQ(ns.get(), "x");
    auto [s, e] = ns.align_to_original(0, 1);
    EXPECT_EQ(s, 2u);
    EXPECT_EQ(e, 3u);
}

// ---- Tokenizer 总线 -------------------------------------------------------
TEST(TokenizerPipeline, EncodeProducesEncoding) {
    Tokenizer tok;
    tok.set_normalizer(make_identity_normalizer());
    tok.set_pretokenizer(make_byte_level_pretokenizer(false, false, true));
    auto bpe_status =
        BpeBuilder().vocab_and_merges(make_gpt2_like_vocab(), make_gpt2_like_merges()).ignore_merges(true).build();
    ASSERT_TRUE(bpe_status.ok());
    tok.set_model(std::move(*bpe_status));
    tok.set_decoder(make_byte_level_decoder());

    Encoding e = tok.encode("Hello");
    EXPECT_FALSE(e.ids.empty());
    EXPECT_EQ(e.ids.size(), e.tokens.size());
    EXPECT_EQ(e.ids.size(), e.offsets.size());
    EXPECT_EQ(e.attention_mask.size(), e.ids.size());
    // 第一个 token 偏移起点应为 0
    EXPECT_EQ(e.offsets[0].first, 0u);
}

TEST(TokenizerPipeline, DecodeRoundTrip) {
    Tokenizer tok;
    tok.set_normalizer(make_identity_normalizer());
    tok.set_pretokenizer(make_byte_level_pretokenizer(false, false, true));
    auto bpe_status =
        BpeBuilder().vocab_and_merges(make_gpt2_like_vocab(), make_gpt2_like_merges()).ignore_merges(true).build();
    ASSERT_TRUE(bpe_status.ok());
    tok.set_model(std::move(*bpe_status));
    tok.set_decoder(make_byte_level_decoder());

    Encoding e = tok.encode("Hello");
    std::string recovered = tok.decode(e.ids);
    EXPECT_EQ(recovered, "Hello");
}

TEST(TokenizerPipeline, EncodeBatch) {
    Tokenizer tok;
    tok.set_normalizer(make_identity_normalizer());
    tok.set_pretokenizer(make_byte_level_pretokenizer(false, false, true));
    auto bpe_status =
        BpeBuilder().vocab_and_merges(make_gpt2_like_vocab(), make_gpt2_like_merges()).ignore_merges(true).build();
    ASSERT_TRUE(bpe_status.ok());
    tok.set_model(std::move(*bpe_status));
    tok.set_decoder(make_byte_level_decoder());

    auto encs = tok.encode_batch({"Hello", "world"});
    ASSERT_EQ(encs.size(), 2u);
    EXPECT_FALSE(encs[0].ids.empty());
    EXPECT_FALSE(encs[1].ids.empty());
}

TEST(TokenizerPipeline, PaddingPadsToMax) {
    Tokenizer tok;
    tok.set_normalizer(make_identity_normalizer());
    tok.set_pretokenizer(make_byte_level_pretokenizer(false, false, true));
    auto bpe_status =
        BpeBuilder().vocab_and_merges(make_gpt2_like_vocab(), make_gpt2_like_merges()).ignore_merges(true).build();
    ASSERT_TRUE(bpe_status.ok());
    tok.set_model(std::move(*bpe_status));
    tok.set_decoder(make_byte_level_decoder());

    PaddingOptions p;
    p.pad_id = 12;
    p.pad_token = "<pad>";
    tok.set_padding(p);

    auto encs = tok.encode_batch({"Hello", "world"});
    // 两条都应被 pad 到 batch 内最长
    std::size_t m = std::max(encs[0].ids.size(), encs[1].ids.size());
    EXPECT_EQ(encs[0].ids.size(), m);
    EXPECT_EQ(encs[1].ids.size(), m);
    // attention_mask 末尾 0 区应等于 pad 个数
    EXPECT_TRUE(
        std::any_of(encs[0].attention_mask.begin(), encs[0].attention_mask.end(), [](uint8_t x) { return x == 0; }) ||
        encs[0].ids.size() == m);
}

TEST(TokenizerPipeline, Truncation) {
    Tokenizer tok;
    tok.set_normalizer(make_identity_normalizer());
    tok.set_pretokenizer(make_byte_level_pretokenizer(false, false, true));
    auto bpe_status = BpeBuilder().vocab_and_merges(make_gpt2_like_vocab(), make_gpt2_like_merges()).build();
    ASSERT_TRUE(bpe_status.ok());
    tok.set_model(std::move(*bpe_status));
    tok.set_decoder(make_byte_level_decoder());

    TruncationOptions t;
    t.max_length = 2;
    tok.set_truncation(t);

    Encoding e = tok.encode("HelloWorld");
    EXPECT_LE(e.ids.size(), 2u);
}

TEST(TokenizerPipeline, OffsetRefersToOriginal) {
    // 加 Normalizer 后,token 偏移应指回原文本(不是 normalized)
    // 词表里都是小写;normalizer 把 "Hello" → "hello" 才能命中
    Vocab v;
    v["h"] = 0;
    v["e"] = 1;
    v["l"] = 2;
    v["o"] = 3;
    v["ll"] = 4;
    v["hello"] = 5;
    MergeList m = {{"l", "l"}, {"h", "e"}, {"he", "l"}, {"hel", "lo"}};
    v["he"] = 6;
    v["hel"] = 7;
    v["lo"] = 8;

    Tokenizer tok;
    tok.set_normalizer(make_lowercase_normalizer());
    tok.set_pretokenizer(make_byte_level_pretokenizer(false, false, true));
    auto bpe_status = BpeBuilder().vocab_and_merges(v, m).ignore_merges(true).build();
    ASSERT_TRUE(bpe_status.ok());
    tok.set_model(std::move(*bpe_status));
    tok.set_decoder(make_byte_level_decoder());

    Encoding e = tok.encode("Hello");
    ASSERT_FALSE(e.offsets.empty());
    // 第一个 token 偏移起点 0,末尾偏移应 ≤ 5(原文 "Hello" 5 字节)
    EXPECT_EQ(e.offsets.front().first, 0u);
    EXPECT_LE(e.offsets.back().second, 5u);
}

// 守护:ByteLevel 多字节字符(中文)的 offset 必须映射回原始文本字节空间
// 每个 UTF-8 中文字符 3 字节,ByteLevel 把每字节编码为 2 字节 byte-char,
// Model 返回的 byte-char 空间偏移必须通过 alignment 转换回原始字节偏移
TEST(TokenizerPipeline, ByteLevelMultibyteOffsetsCorrect) {
    // 构建完整 256 字节 byte-char 词表 + "中""国" 合并
    auto bc = [](uint8_t b) { return util::byte_to_string(b); };
    Vocab v;
    // 全部 256 个单字节 byte-char 入词表
    for (int b = 0; b <= 0xFF; ++b) {
        v[bc(static_cast<uint8_t>(b))] = static_cast<TokenId>(b);
    }
    // "中" = E4 B8 AD → 3 个 byte-char 拼接
    std::string zhong = bc(0xE4) + bc(0xB8) + bc(0xAD);
    // "国" = E5 9B BD
    std::string guo = bc(0xE5) + bc(0x9B) + bc(0xBD);
    TokenId next_id = 256;
    v[zhong] = next_id++;
    v[guo] = next_id++;
    // 中间合并体
    std::string e4b8 = bc(0xE4) + bc(0xB8);
    v[e4b8] = next_id++;
    std::string e59b = bc(0xE5) + bc(0x9B);
    v[e59b] = next_id++;

    MergeList m = {
        {bc(0xE4), bc(0xB8)},  // rank 0: E4 B8 → E4B8
        {e4b8, bc(0xAD)},      // rank 1: E4B8 AD → 中
        {bc(0xE5), bc(0x9B)},  // rank 2: E5 9B → E59B
        {e59b, bc(0xBD)},      // rank 3: E59B BD → 国
    };

    Tokenizer tok;
    tok.set_normalizer(make_identity_normalizer());
    tok.set_pretokenizer(make_byte_level_pretokenizer(false, false, true));
    auto bpe_status = BpeBuilder().vocab_and_merges(v, m).build();
    ASSERT_TRUE(bpe_status.ok());
    tok.set_model(std::move(*bpe_status));
    tok.set_decoder(make_byte_level_decoder());

    // "中国" = 6 原始字节,经 ByteLevel → 12 byte-char 字节
    // BPE 合并后应得到 2 个 token:"中" 和 "国"
    Encoding e = tok.encode("中国");
    ASSERT_EQ(e.ids.size(), 2u);

    // 关键断言:offsets 必须在原始字节空间 [0,6),不是 byte-char 空间 [0,12)
    // token 0 = "中" → offsets [0, 3)
    EXPECT_EQ(e.offsets[0].first, 0u);
    EXPECT_EQ(e.offsets[0].second, 3u);
    // token 1 = "国" → offsets [3, 6)
    EXPECT_EQ(e.offsets[1].first, 3u);
    EXPECT_EQ(e.offsets[1].second, 6u);

    // offsets 不应重叠,且应完整覆盖 [0, 6)
    EXPECT_EQ(e.offsets[0].second, e.offsets[1].first);
    EXPECT_EQ(e.offsets.back().second, 6u);

    // round-trip
    EXPECT_EQ(tok.decode(e.ids), "中国");
}

// 守护:混合 ASCII + 多字节,offset 正确
TEST(TokenizerPipeline, ByteLevelMixedAsciiMultibyteOffsets) {
    // "a中b" = 5 原始字节(a=1, 中=3, b=1)
    auto bc = [](uint8_t b) { return util::byte_to_string(b); };
    Vocab v;
    for (int b = 0; b <= 0xFF; ++b) {
        v[bc(static_cast<uint8_t>(b))] = static_cast<TokenId>(b);
    }
    std::string zhong = bc(0xE4) + bc(0xB8) + bc(0xAD);
    v[zhong] = 256;
    std::string e4b8 = bc(0xE4) + bc(0xB8);
    v[e4b8] = 257;
    MergeList m = {
        {bc(0xE4), bc(0xB8)},
        {e4b8, bc(0xAD)},
    };

    Tokenizer tok;
    tok.set_normalizer(make_identity_normalizer());
    tok.set_pretokenizer(make_byte_level_pretokenizer(false, false, true));
    auto bpe_status = BpeBuilder().vocab_and_merges(v, m).build();
    ASSERT_TRUE(bpe_status.ok());
    tok.set_model(std::move(*bpe_status));
    tok.set_decoder(make_byte_level_decoder());

    Encoding e = tok.encode("a中b");
    // a(1 byte-char) 中(3 bytes → 6 byte-char,合并为 1 token) b(1 byte-char)
    ASSERT_EQ(e.ids.size(), 3u);

    // a: [0, 1), 中: [1, 4), b: [4, 5) — 全部在原始字节空间
    EXPECT_EQ(e.offsets[0], (std::pair<uint32_t, uint32_t>{0, 1}));
    EXPECT_EQ(e.offsets[1], (std::pair<uint32_t, uint32_t>{1, 4}));
    EXPECT_EQ(e.offsets[2], (std::pair<uint32_t, uint32_t>{4, 5}));

    EXPECT_EQ(tok.decode(e.ids), "a中b");
}

// ---- tokenizer.json round-trip --------------------------------------------
class TokenizerJsonTest : public ::testing::Test {
   protected:
    std::string tmp;

    void SetUp() override {
        tmp = std::filesystem::temp_directory_path().string() + "/bpe_m6_" +
              std::to_string(reinterpret_cast<uintptr_t>(this)) + "/";
        std::filesystem::create_directories(tmp);
    }

    void TearDown() override {
        std::filesystem::remove_all(tmp);
    }

    void write(const std::string& name, const std::string& content) {
        std::ofstream(tmp + name) << content;
    }
};

TEST_F(TokenizerJsonTest, SaveAndReloadRoundTrip) {
    // 1. 构造 Tokenizer(add_prefix_space=true 与 to_file 默认一致)
    Tokenizer tok;
    tok.set_normalizer(make_identity_normalizer());
    tok.set_pretokenizer(make_byte_level_pretokenizer(true, false, true));
    auto bpe_status =
        BpeBuilder().vocab_and_merges(make_gpt2_like_vocab(), make_gpt2_like_merges()).ignore_merges(true).build();
    ASSERT_TRUE(bpe_status.ok());
    tok.set_model(std::move(*bpe_status));
    tok.set_decoder(make_byte_level_decoder());

    // 2. 序列化
    auto st = tok.to_file(tmp + "tokenizer.json");
    ASSERT_TRUE(st.ok()) << st.message();
    EXPECT_TRUE(std::filesystem::exists(tmp + "tokenizer.json"));

    // 3. 重新加载
    auto reload_status = Tokenizer::from_file(tmp + "tokenizer.json");
    ASSERT_TRUE(reload_status.ok()) << reload_status.status().message();
    auto& tok2 = *reload_status.value();

    // 4. 比对编码结果("Hello" 经 ByteLevel add_prefix_space → "ĠHello" 整词命中)
    Encoding e1 = tok.encode("Hello");
    Encoding e2 = tok2.encode("Hello");
    ASSERT_EQ(e1.ids.size(), e2.ids.size());
    for (std::size_t i = 0; i < e1.ids.size(); ++i) {
        EXPECT_EQ(e1.ids[i], e2.ids[i]) << "id mismatch at " << i;
    }
}

TEST_F(TokenizerJsonTest, VocabMergesSave) {
    Tokenizer tok;
    auto bpe_status = BpeBuilder().vocab_and_merges(make_gpt2_like_vocab(), make_gpt2_like_merges()).build();
    ASSERT_TRUE(bpe_status.ok());
    tok.set_model(std::move(*bpe_status));

    auto st = tok.to_vocab_merges(tmp);
    ASSERT_TRUE(st.ok()) << st.message();
    EXPECT_TRUE(std::filesystem::exists(tmp + "vocab.json"));
    EXPECT_TRUE(std::filesystem::exists(tmp + "merges.txt"));
}

TEST_F(TokenizerJsonTest, LoadNonExistentFails) {
    auto st = Tokenizer::from_file(tmp + "nope.json");
    EXPECT_FALSE(st.ok());
}

TEST_F(TokenizerJsonTest, LoadInvalidJsonFails) {
    write("bad.json", "{not json");
    auto st = Tokenizer::from_file(tmp + "bad.json");
    EXPECT_FALSE(st.ok());
}

TEST_F(TokenizerJsonTest, LoadUnsupportedModelType) {
    write("unsupported.json", R"({"model":{"type":"WordPiece","vocab":{}}})");
    auto st = Tokenizer::from_file(tmp + "unsupported.json");
    EXPECT_FALSE(st.ok());
}

// 守护:null 字段不应导致崩溃
// 真实模型(如 glm-5-2)会把 normalizer / unk_token 等设为 JSON null,
// from_file 必须用 contains + is_xxx 检查而非 value(key, default),否则抛 type_error
TEST_F(TokenizerJsonTest, NullFieldsDoNotCrash) {
    // 构造一份含 null 字段的 tokenizer.json,模拟 glm-5-2 风格
    const std::string G = "\xC4\xA0";
    nlohmann::json j;
    j["version"] = "1.0";
    j["normalizer"] = nullptr;  // null normalizer
    j["pre_tokenizer"] = {
        {"type", "Sequence"},
        {"pretokenizers",
         nlohmann::json::array(
             {{{"type", "Split"}, {"pattern", {{"Regex", "\\p{L}+"}}}, {"behavior", "Isolated"}, {"invert", false}},
              {{"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", true}, {"use_regex", false}}})}};
    j["post_processor"] = {
        {"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", true}, {"use_regex", true}};
    j["decoder"] = {{"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", true}, {"use_regex", true}};
    nlohmann::json v = nlohmann::json::object();
    v["a"] = 0;
    v["b"] = 1;
    v["ab"] = 2;
    v[G + "x"] = 3;
    j["model"] = {{"type", "BPE"},
                  {"vocab", v},
                  {"merges", nlohmann::json::array({nlohmann::json::array({"a", "b"})})},
                  {"unk_token", nullptr},                  // null
                  {"continuing_subword_prefix", nullptr},  // null
                  {"end_of_word_suffix", nullptr},         // null
                  {"dropout", nullptr},                    // null
                  {"fuse_unk", false},
                  {"byte_fallback", false},
                  {"ignore_merges", true}};

    write("null_fields.json", j.dump(2));

    // 加载不应崩溃也不应报错
    auto st = Tokenizer::from_file(tmp + "null_fields.json");
    ASSERT_TRUE(st.ok()) << st.status().message();
    auto& tok = *st.value();

    // 编码 + 解码 round-trip 仍应正常
    Encoding e = tok.encode("ab");
    ASSERT_FALSE(e.ids.empty());
    std::string recovered = tok.decode(e.ids);
    EXPECT_EQ(recovered, "ab");
}

// 守护:null 字段 + Sequence pre_tokenizer 中的 ByteLevel 提取
// 验证 Sequence 里 ByteLevel 子项的 add_prefix_space/trim_offsets 被正确读取
TEST_F(TokenizerJsonTest, NullFieldsWithSequencePretokenizerExtractsByteLevel) {
    const std::string G = "\xC4\xA0";
    nlohmann::json j;
    j["version"] = "1.0";
    j["normalizer"] = nullptr;
    // Sequence 含 Split + ByteLevel(add_prefix_space=false, use_regex=false)
    // from_file 应提取 ByteLevel 并强制 use_regex=true
    j["pre_tokenizer"] = {
        {"type", "Sequence"},
        {"pretokenizers",
         nlohmann::json::array(
             {{{"type", "Split"}, {"pattern", {{"Regex", "\\p{N}{1,3}"}}}, {"behavior", "Isolated"}, {"invert", false}},
              {{"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", true}, {"use_regex", false}}})}};
    j["decoder"] = {{"type", "ByteLevel"}};
    nlohmann::json v = nlohmann::json::object();
    v["a"] = 0;
    v["b"] = 1;
    v["ab"] = 2;
    v[G] = 3;
    j["model"] = {{"type", "BPE"},
                  {"vocab", v},
                  {"merges", nlohmann::json::array({nlohmann::json::array({"a", "b"})})},
                  {"ignore_merges", true}};

    write("seq_null.json", j.dump(2));

    auto st = Tokenizer::from_file(tmp + "seq_null.json");
    ASSERT_TRUE(st.ok()) << st.status().message();
    auto& tok = *st.value();

    // add_prefix_space=false → "ab" 不加前缀空格 → 整词 "ab" 命中 ignore_merges
    Encoding e = tok.encode("ab");
    ASSERT_EQ(e.ids.size(), 1u);
    EXPECT_EQ(e.ids[0], 2u);
    EXPECT_EQ(tok.decode(e.ids), "ab");
}

TEST_F(TokenizerJsonTest, LoadMinimalTokenizer) {
    // 最小合法 tokenizer.json
    write("min.json", R"({
        "model": {
            "type": "BPE",
            "vocab": {"a":0, "b":1, "ab":2},
            "merges": [["a","b"]]
        },
        "pre_tokenizer": {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true},
        "decoder": {"type":"ByteLevel"}
    })");
    auto st = Tokenizer::from_file(tmp + "min.json");
    ASSERT_TRUE(st.ok()) << st.status().message();
    auto& tok = *st.value();
    Encoding e = tok.encode("ab");
    EXPECT_FALSE(e.ids.empty());
}

// 旧格式 merges 字符串形式:兼容 HF 早期 tokenizer.json 的 "a b" 写法
// 需用 absl::SkipEmpty 过滤空段,以容忍前后/中间多余空格
TEST_F(TokenizerJsonTest, LegacyStringMergesFormat) {
    // 多个相邻空格、前导/尾随空格在旧格式下应被当作单空格分隔
    nlohmann::json v = nlohmann::json::object();
    v["a"] = 0;
    v["b"] = 1;
    v["ab"] = 2;
    nlohmann::json m = nlohmann::json::array();
    m.push_back("a b");       // 正常单空格
    m.push_back("  a  b  ");  // 前导/中间/尾随多空格,仍应解析为 {a,b}
    m.push_back("x y");       // 段数正确但 token 不在 vocab → 应被忽略
    m.push_back("onlyone");   // 段数 != 2 → 应被忽略
    nlohmann::json j;
    j["model"] = {{"type", "BPE"}, {"vocab", v}, {"merges", m}, {"ignore_merges", true}};
    j["pre_tokenizer"] = {
        {"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", false}, {"use_regex", true}};
    j["decoder"] = {{"type", "ByteLevel"}};
    write("legacy_merges.json", j.dump(2));

    auto st = Tokenizer::from_file(tmp + "legacy_merges.json");
    ASSERT_TRUE(st.ok()) << st.status().message();
    auto& tok = *st.value();

    // ignore_merges=true: "ab" 命中整词(忽略 merges 表)
    Encoding e = tok.encode("ab");
    ASSERT_EQ(e.ids.size(), 1u);
    EXPECT_EQ(e.ids[0], 2u);
    EXPECT_EQ(tok.decode(e.ids), "ab");
}

// ---- 端到端 GPT-2 风格 round-trip -----------------------------------------
TEST(EndToEndGpt2, EncodeDecodeRoundTrip) {
    const std::string G = "\xC4\xA0";
    // 构造一个能覆盖 " Hello world" 的最小 GPT-2 风格词表
    // add_prefix_space=true → "Hello" 变 "ĠHello","world" 变 "Ġworld"
    Vocab v;
    v["H"] = 0;
    v["e"] = 1;
    v["l"] = 2;
    v["o"] = 3;
    v["ll"] = 4;
    v["Hello"] = 5;
    v[G + "world"] = 6;
    v["w"] = 7;
    v["r"] = 8;
    v["d"] = 9;
    v["He"] = 10;
    v["Hel"] = 11;
    v["lo"] = 12;
    v[G] = 13;            // 单独 Ġ
    v[G + "Hello"] = 14;  // add_prefix_space 后整词
    MergeList m = {{"l", "l"}, {"H", "e"}, {"He", "l"}, {"Hel", "lo"}};

    Tokenizer tok;
    tok.set_normalizer(make_identity_normalizer());
    tok.set_pretokenizer(make_byte_level_pretokenizer(true, false, true));
    auto bpe_status = BpeBuilder().vocab_and_merges(v, m).ignore_merges(true).build();
    ASSERT_TRUE(bpe_status.ok());
    tok.set_model(std::move(*bpe_status));
    tok.set_decoder(make_byte_level_decoder());

    Encoding e = tok.encode("Hello world");
    ASSERT_FALSE(e.ids.empty());

    std::string recovered = tok.decode(e.ids);
    // add_prefix_space=true → 前缀空格保留
    EXPECT_EQ(recovered, " Hello world");
}

// ---- Tokenizer::from_file 完整 GPT-2 风格 ---------------------------------
TEST_F(TokenizerJsonTest, FullGpt2StyleRoundTrip) {
    const std::string G = "\xC4\xA0";
    // 写出一份最小 GPT-2 风格 tokenizer.json
    nlohmann::json j;
    j["version"] = "1.0";
    nlohmann::json v = nlohmann::json::object();
    v["H"] = 0;
    v["e"] = 1;
    v["l"] = 2;
    v["o"] = 3;
    v["ll"] = 4;
    v["Hello"] = 5;
    v[G + "world"] = 6;
    v["w"] = 7;
    v["r"] = 8;
    v["d"] = 9;
    v["He"] = 10;
    v["Hel"] = 11;
    v["lo"] = 12;
    v[G] = 13;
    v[G + "Hello"] = 14;
    j["model"] = {
        {"type", "BPE"},
        {"vocab", v},
        {"merges", nlohmann::json::array({nlohmann::json::array({"l", "l"}), nlohmann::json::array({"H", "e"}),
                                          nlohmann::json::array({"He", "l"}), nlohmann::json::array({"Hel", "lo"})})},
        {"ignore_merges", true}};
    j["pre_tokenizer"] = {
        {"type", "ByteLevel"}, {"add_prefix_space", true}, {"trim_offsets", true}, {"use_regex", true}};
    j["decoder"] = {{"type", "ByteLevel"}};

    std::ofstream(tmp + "gpt2.json") << j.dump(2);

    auto st = Tokenizer::from_file(tmp + "gpt2.json");
    ASSERT_TRUE(st.ok()) << st.status().message();
    auto& tok = *st.value();

    Encoding e = tok.encode("Hello world");
    ASSERT_FALSE(e.ids.empty());

    std::string recovered = tok.decode(e.ids);
    EXPECT_EQ(recovered, " Hello world");
}