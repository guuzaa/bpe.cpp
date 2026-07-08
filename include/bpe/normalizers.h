// bpe/normalizers.h — Normalizer 工厂
#pragma once

#include <memory>

#include "bpe/interfaces.h"

namespace bpe {

// Identity normalizer:不改变字符串
std::unique_ptr<Normalizer> make_identity_normalizer();

// Strip:仅去首尾空白字节(不改变中间)
std::unique_ptr<Normalizer> make_strip_normalizer();

// Lowercase / Uppercase:ASCII + 拉丁 1 段大小写转换
std::unique_ptr<Normalizer> make_lowercase_normalizer();
std::unique_ptr<Normalizer> make_uppercase_normalizer();

// 顺序组合多个 normalizer
std::unique_ptr<Normalizer> make_sequence_normalizer(std::vector<std::unique_ptr<Normalizer>> ns);

}  // namespace bpe