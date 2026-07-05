// tests/test_trainer.cpp — M4: BpeTrainer 训练流程
//
// 覆盖:
//   - 未喂语料 → train 报错
//   - 喂语料 + train → vocab/merges 非空
//   - 简单词表训练:对 "ab ab ab" 训练,得到 merge (a,b)→ab
//   - 端到端:训完后用 BPE.tokenize 验证一致
//   - special_tokens 纳入 vocab
//   - end_of_word_suffix:训练后 encode 用到 </w>
#include <gtest/gtest.h>
#include <numeric>
#include <string>

#include "bpe/bpe.h"
#include "bpe/interfaces.h"
#include "bpe/trainer.h"
#include "pretokenizers/byte_level.h"

using namespace bpe;

namespace {

// 简单空白 pre-tokenizer:按空格切分(单字符级调试用)
class WhitespaceSplit : public PreTokenizer {
   public:
    void pre_tokenize(PreTokenizedString& s) const override {
        auto& pts = s.tokens();
        std::vector<PreToken> out;
        for (const auto& pt : pts) {
            const std::string& text = pt.text;
            std::size_t i = 0;
            while (i < text.size()) {
                while (i < text.size() && text[i] == ' ') {
                    ++i;
                }
                std::size_t j = i;
                while (j < text.size() && text[j] != ' ') {
                    ++j;
                }
                if (j > i) {
                    uint32_t off = pt.offsets.first + static_cast<uint32_t>(i);
                    out.push_back({text.substr(i, j - i), {off, off + static_cast<uint32_t>(j - i)}, {}});
                }
                i = j;
            }
        }
        pts = std::move(out);
    }
};

// 一个最简单的 pre-tokenizer:整段当一个 token
class IdentityPreTok : public PreTokenizer {
   public:
    void pre_tokenize(PreTokenizedString& s) const override {
        (void)s;
    }
};

}  // namespace

// ---- 基础:未喂语料 -------------------------------------------------------
TEST(BpeTrainer, EmptyCorpusFails) {
    auto bpe_status = BpeBuilder::build_default();
    ASSERT_TRUE(bpe_status.ok());
    BpeTrainer trainer;
    auto status = trainer.train(*bpe_status.value());
    EXPECT_FALSE(status.ok());
}

// ---- 基础:训练产出 vocab --------------------------------------------------
TEST(BpeTrainer, ProducesVocabAndMerges) {
    auto bpe_status = BpeBuilder::build_default();
    ASSERT_TRUE(bpe_status.ok());
    auto& bpe = *bpe_status.value();

    BpeTrainerOptions opts;
    opts.vocab_size = 100;
    opts.min_frequency = 1;
    WhitespaceSplit pretok;
    BpeTrainer trainer(opts);
    trainer.feed({"ab ab ab"}, pretok);
    EXPECT_GT(trainer.word_count(), 0u);
    auto status = trainer.train(bpe);
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_GE(bpe.vocab_size(), 2u);  // 至少含 a 和 b
}

// ---- 简单合并:对 "ab ab ab ab" 训练,得到 (a,b)→ab ------------------------
TEST(BpeTrainer, LearnsMergeAB) {
    auto bpe_status = BpeBuilder::build_default();
    ASSERT_TRUE(bpe_status.ok());
    auto& bpe = *bpe_status.value();

    BpeTrainerOptions opts;
    opts.vocab_size = 100;
    opts.min_frequency = 1;  // 4 次出现,a-b 邻接 4 次
    WhitespaceSplit pretok;
    BpeTrainer trainer(opts);
    trainer.feed({"ab ab ab ab"}, pretok);
    ASSERT_TRUE(trainer.train(bpe).ok());

    // 训完后,"ab" 应当是一个 token
    auto ab_id = bpe.token_to_id("ab");
    ASSERT_TRUE(ab_id.has_value()) << "expected 'ab' in vocab after training";

    // 编码 "ab" 应得到单 token
    auto tokens = bpe.tokenize("ab");
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0].id, *ab_id);
    EXPECT_EQ(tokens[0].value, "ab");
}

// ---- min_frequency 过滤:pair 出现不足不合并 -------------------------------
TEST(BpeTrainer, MinFrequencyFilters) {
    auto bpe_status = BpeBuilder::build_default();
    ASSERT_TRUE(bpe_status.ok());
    auto& bpe = *bpe_status.value();

    BpeTrainerOptions opts;
    opts.vocab_size = 100;
    opts.min_frequency = 5;  // (a,b) 出现 4 次 < 5 → 不合并
    WhitespaceSplit pretok;
    BpeTrainer trainer(opts);
    trainer.feed({"ab ab ab ab"}, pretok);
    ASSERT_TRUE(trainer.train(bpe).ok());

    // "ab" 不应在 vocab 中(作为合并 token)
    EXPECT_FALSE(bpe.token_to_id("ab").has_value());
    // 但 a、b 在(作为字母表)
    EXPECT_TRUE(bpe.token_to_id("a").has_value());
    EXPECT_TRUE(bpe.token_to_id("b").has_value());

    // encode "ab" 应得到 2 个 token
    auto tokens = bpe.tokenize("ab");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0].value, "a");
    EXPECT_EQ(tokens[1].value, "b");
}

// ---- special_tokens 写入 vocab --------------------------------------------
TEST(BpeTrainer, SpecialTokensInVocab) {
    auto bpe_status = BpeBuilder::build_default();
    ASSERT_TRUE(bpe_status.ok());
    auto& bpe = *bpe_status.value();

    BpeTrainerOptions opts;
    opts.vocab_size = 100;
    opts.min_frequency = 1;
    opts.special_tokens = {"<unk>", "<pad>"};
    WhitespaceSplit pretok;
    BpeTrainer trainer(opts);
    trainer.feed({"ab"}, pretok);
    ASSERT_TRUE(trainer.train(bpe).ok());
    EXPECT_TRUE(bpe.token_to_id("<unk>").has_value());
    EXPECT_TRUE(bpe.token_to_id("<pad>").has_value());
}

// ---- initial_alphabet 强制纳入罕见字符 ------------------------------------
TEST(BpeTrainer, InitialAlphabetForced) {
    auto bpe_status = BpeBuilder::build_default();
    ASSERT_TRUE(bpe_status.ok());
    auto& bpe = *bpe_status.value();

    BpeTrainerOptions opts;
    opts.vocab_size = 100;
    opts.min_frequency = 1;
    opts.initial_alphabet = {"z"};  // z 在语料中未出现,但强制纳入
    WhitespaceSplit pretok;
    BpeTrainer trainer(opts);
    trainer.feed({"ab ab"}, pretok);
    ASSERT_TRUE(trainer.train(bpe).ok());
    EXPECT_TRUE(bpe.token_to_id("z").has_value());
}

// ---- limit_alphabet 裁剪 ---------------------------------------------------
TEST(BpeTrainer, LimitAlphabetTrims) {
    auto bpe_status = BpeBuilder::build_default();
    ASSERT_TRUE(bpe_status.ok());
    auto& bpe = *bpe_status.value();

    BpeTrainerOptions opts;
    opts.vocab_size = 100;
    opts.min_frequency = 1;
    opts.limit_alphabet = 2;  // 只保留出现最多的 2 个字符
    WhitespaceSplit pretok;
    BpeTrainer trainer(opts);
    // 语料:a 出现 4 次,b 出现 4 次,c 出现 2 次;d 出现 1 次
    trainer.feed({"ab ab cd abcd"}, pretok);
    ASSERT_TRUE(trainer.train(bpe).ok());
    EXPECT_TRUE(bpe.token_to_id("a").has_value());
    EXPECT_TRUE(bpe.token_to_id("b").has_value());
    EXPECT_FALSE(bpe.token_to_id("c").has_value());  // 第 3 位被裁
    EXPECT_FALSE(bpe.token_to_id("d").has_value());
}

// ---- vocab_size 上限:达到目标即停止 --------------------------------------
TEST(BpeTrainer, VocabSizeStopsMerging) {
    auto bpe_status = BpeBuilder::build_default();
    ASSERT_TRUE(bpe_status.ok());
    auto& bpe = *bpe_status.value();

    BpeTrainerOptions opts;
    // 字母表 a, b 即 2 个,目标 vocab_size=3 → 留 1 个名额给一个合并
    opts.vocab_size = 3;
    opts.min_frequency = 1;
    WhitespaceSplit pretok;
    BpeTrainer trainer(opts);
    trainer.feed({"ab ab ab"}, pretok);
    ASSERT_TRUE(trainer.train(bpe).ok());
    EXPECT_EQ(bpe.vocab_size(), 3u);
    EXPECT_TRUE(bpe.token_to_id("ab").has_value());
}

// ---- 端到端 + ByteLevel pretokenizer + 训练 + 编码 ------------------------
TEST(BpeTrainer, EndToEndWithByteLevel) {
    // 语料:几条简单英文,用 ByteLevel(无 regex)切分后训练
    auto bpe_status = BpeBuilder::build_default();
    ASSERT_TRUE(bpe_status.ok());
    auto& bpe = *bpe_status.value();

    pretokenizers::ByteLevel pretok(/*add_prefix_space=*/false,
                                    /*trim_offsets=*/false,
                                    /*use_regex=*/false);
    BpeTrainerOptions opts;
    opts.vocab_size = 100;
    opts.min_frequency = 1;
    BpeTrainer trainer(opts);
    trainer.feed({"hello world hello"}, pretok);
    ASSERT_TRUE(trainer.train(bpe).ok()) << "train failed";

    // 训完后 BF 应该能 token_to_id 单字节 token(都是 ASCII,映射后仍为自身)
    EXPECT_TRUE(bpe.token_to_id("h").has_value());
    EXPECT_TRUE(bpe.token_to_id("e").has_value());

    // encode "hello" 应该产生一些 token,且能 round-trip
    std::string text = "hello";
    std::string mapped = pretokenizers::ByteLevel::encode_bytes(text);
    auto tokens = bpe.tokenize(mapped);
    EXPECT_FALSE(tokens.empty());

    // decode 还原
    std::vector<std::string> tok_strs;
    for (auto& t : tokens) {
        tok_strs.push_back(t.value);
    }
    std::string recovered =
        pretokenizers::ByteLevel::decode_bytes(std::accumulate(tok_strs.begin(), tok_strs.end(), std::string{}));
    EXPECT_EQ(recovered, text);
}

// ---- end_of_word_suffix:训练时 </w> 加在词尾 ------------------------------
TEST(BpeTrainer, EndOfWordSuffix) {
    auto bpe_status = BpeBuilder::build_default();
    ASSERT_TRUE(bpe_status.ok());
    auto& bpe = *bpe_status.value();

    BpeTrainerOptions opts;
    opts.vocab_size = 100;
    opts.min_frequency = 1;
    opts.end_of_word_suffix = "</w>";
    WhitespaceSplit pretok;
    BpeTrainer trainer(opts);
    // 语料:单词 "he" 与 "lo" 各 3 次;词尾加 </w> 后末字符变 e</w>、o</w>
    trainer.feed({"he lo he lo he lo"}, pretok);
    ASSERT_TRUE(trainer.train(bpe).ok()) << "train failed";

    // 检查 vocab 里至少包含末字符的 </w> 后缀版本
    EXPECT_TRUE(bpe.token_to_id("e</w>").has_value());
    EXPECT_TRUE(bpe.token_to_id("o</w>").has_value());

    // 此时 model.end_of_word_suffix 已被 train 写入 "</w>",
    // encode "he" 时末字符 e 应自动拼成 e</w>
    auto tokens = bpe.tokenize("he");
    ASSERT_FALSE(tokens.empty());
    // 应至少有一个 token 的 value 含 "</w>"
    bool has_eow = false;
    for (const auto& t : tokens) {
        if (t.value.find("</w>") != std::string::npos) {
            has_eow = true;
            break;
        }
    }
    EXPECT_TRUE(has_eow);
}

// ---- 进度回调 --------------------------------------------------------------
TEST(BpeTrainer, ProgressCallback) {
    auto bpe_status = BpeBuilder::build_default();
    ASSERT_TRUE(bpe_status.ok());
    auto& bpe = *bpe_status.value();

    std::size_t calls = 0;
    BpeTrainerOptions opts;
    opts.vocab_size = 5;
    opts.min_frequency = 1;
    opts.progress = [&](std::size_t, std::size_t, const char*) { ++calls; };
    WhitespaceSplit pretok;
    BpeTrainer trainer(opts);
    trainer.feed({"ab ab ab ab"}, pretok);
    ASSERT_TRUE(trainer.train(bpe).ok());
    EXPECT_GT(calls, 0u);
}

// ---- 字母表覆盖训练语料全部字符 --------------------------------------------
TEST(BpeTrainer, AlphhabtCoversCorpus) {
    auto bpe_status = BpeBuilder::build_default();
    ASSERT_TRUE(bpe_status.ok());
    auto& bpe = *bpe_status.value();

    BpeTrainerOptions opts;
    opts.vocab_size = 200;
    opts.min_frequency = 1;
    WhitespaceSplit pretok;
    BpeTrainer trainer(opts);
    trainer.feed({"abcd wxyz"}, pretok);
    ASSERT_TRUE(trainer.train(bpe).ok());
    for (char c : std::string("abcdwxyz")) {
        std::string s(1, c);
        EXPECT_TRUE(bpe.token_to_id(s).has_value()) << "char '" << c << "' missing";
    }
}