#include <gtest/gtest.h>

#include "bpe/interfaces.h"
#include "bpe/normalizers.h"

using namespace bpe;

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
