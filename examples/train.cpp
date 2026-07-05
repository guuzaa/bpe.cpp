// examples/train.cpp — 在内置语料上训练 BPE,保存 tokenizer.json,reload 验证
//
// 用法:
//   ./train                        # 用内置语料训练
//   ./train 500 /tmp/my_tok.json   # 指定词表大小和输出路径
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "bpe/bpe.h"
#include "bpe/normalizers.h"
#include "bpe/pretokenizers.h"
#include "bpe/tokenizer.h"
#include "bpe/trainer.h"

namespace {

// 一段足够训练出有意义 merges 的英文语料
const char* kCorpus[] = {
    "the quick brown fox jumps over the lazy dog",
    "hello world hello bpe cpp",
    "byte pair encoding is a subword tokenization algorithm",
    "the the the the the the the the the the",
    "hello hello hello world world world",
    "tokenization is the process of breaking text into tokens",
    "byte pair encoding merges the most frequent pairs",
    "the quick brown fox and the lazy dog are friends",
    "subword tokenization is useful for many nlp tasks",
    "bpe cpp is a c plus plus implementation of byte pair encoding",
};

}  // namespace

int main(int argc, char** argv) {
    std::size_t vocab_size = 300;
    std::string out_path = "assets/trained-tokenizer.json";

    if (argc >= 2) {
        vocab_size = static_cast<std::size_t>(std::atoi(argv[1]));
    }
    if (argc >= 3) {
        out_path = argv[2];
    }

    // 1. 构造一个空的 BPE 模型 + ByteLevel pre-tokenizer
    auto bpe_status = bpe::BpeBuilder::build_default();
    if (!bpe_status.ok()) {
        std::cerr << "Failed to create BPE: " << bpe_status.status().message() << "\n";
        return 1;
    }
    auto bpe = std::move(*bpe_status);

    // 2. 配置训练器
    bpe::BpeTrainerOptions opts;
    opts.vocab_size = vocab_size;
    opts.min_frequency = 1;
    opts.special_tokens = {"<unk>", "<pad>"};

    bpe::BpeTrainer trainer(opts);

    // 3. 喂语料(使用 ByteLevel pre-tokenizer,add_prefix_space=false,use_regex=true)
    auto pretok = bpe::make_byte_level_pretokenizer(false, false, true);
    std::vector<std::string> corpus(std::begin(kCorpus), std::end(kCorpus));

    std::cerr << "Feeding " << corpus.size() << " lines ...\n";
    trainer.feed(corpus, *pretok);
    std::cerr << "Unique words: " << trainer.word_count() << "\n";

    // 4. 训练
    auto t0 = std::chrono::steady_clock::now();
    auto status = trainer.train(*bpe);
    auto t1 = std::chrono::steady_clock::now();
    if (!status.ok()) {
        std::cerr << "Training failed: " << status.message() << "\n";
        return 1;
    }
    auto train_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cerr << "Trained vocab_size=" << bpe->vocab_size() << " in " << train_ms << " ms\n";

    // 5. 组装 Tokenizer 并保存
    bpe::Tokenizer tok;
    tok.set_normalizer(bpe::make_identity_normalizer());
    tok.set_pretokenizer(bpe::make_byte_level_pretokenizer(false, false, true));
    tok.set_model(std::move(bpe));
    tok.set_decoder(bpe::make_byte_level_decoder());

    auto save_st = tok.to_file(out_path);
    if (!save_st.ok()) {
        std::cerr << "Save failed: " << save_st.message() << "\n";
        return 1;
    }
    std::cerr << "Saved to " << out_path << "\n";

    // 6. 也保存 vocab.json + merges.txt
    std::filesystem::path dir = std::filesystem::path(out_path).parent_path();
    if (dir.empty()) {
        dir = ".";
    }
    auto vm_st = tok.to_vocab_merges(dir.string() + "/trained");
    if (vm_st.ok()) {
        std::cerr << "Saved vocab.json + merges.txt to " << dir / "trained" << "\n";
    }

    // 7. 重新加载并验证 round-trip
    std::cerr << "\nReloading and verifying ...\n";
    auto reload_st = bpe::Tokenizer::from_file(out_path);
    if (!reload_st.ok()) {
        std::cerr << "Reload failed: " << reload_st.status().message() << "\n";
        return 1;
    }
    auto& tok2 = *reload_st.value();

    std::string test = "hello world";
    auto e1 = tok.encode(test);
    auto e2 = tok2.encode(test);
    if (e1.ids == e2.ids) {
        std::printf("Reload round-trip: OK (\"%s\" → %zu tokens)\n", test.c_str(), e1.ids.size());
    } else {
        std::cerr << "Reload round-trip: MISMATCH\n";
        return 1;
    }

    // 8. 显示几个 encode 结果
    std::printf("\nDemo encode:\n");
    for (const char* s : {"hello", "world", "the quick brown fox", "tokenization"}) {
        auto enc = tok.encode(s);
        std::string recovered = tok.decode(enc.ids);
        std::printf("  \"%s\" → %zu tokens → \"%s\"%s\n", s, enc.ids.size(), recovered.c_str(),
                    recovered == s ? " (OK)" : " (MISMATCH)");
    }

    return 0;
}
