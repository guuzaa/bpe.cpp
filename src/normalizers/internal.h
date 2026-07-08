// normalizers/internal.h — 仅库内使用的 normalizer 辅助接口
#pragma once

#include <memory>
#include <string_view>

#include "bpe/interfaces.h"

namespace bpe {

enum class NormalizerKind {
    kIdentity,
    kStrip,      // 去首尾空白
    kLowercase,  // ASCII / 基本多语言平面 tolower(不依赖 ICU)
    kUppercase,
};

// 字符串 type(如 "Lowercase")→ NormalizerKind;未识别(含 "Sequence")→ kIdentity
NormalizerKind parse_normalizer_kind(std::string_view type) noexcept;

// 按 kind 构造单个 normalizer
std::unique_ptr<Normalizer> make_normalizer(NormalizerKind kind);

}  // namespace bpe