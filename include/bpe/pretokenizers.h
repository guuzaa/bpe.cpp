// bpe/pretokenizers.h — PreTokenizer 工厂与对外枚举
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "bpe/interfaces.h"

namespace bpe {

enum class PreTokenizerKind {
    kByteLevel,
    kWhitespace,
    kPunctuation,
};

// ByteLevel 工厂:add_prefix_space / trim_offsets / use_regex
std::unique_ptr<PreTokenizer> make_byte_level_pretokenizer(bool add_prefix_space = true, bool trim_offsets = true,
                                                           bool use_regex = true);

// ---- Split ----------------------------------------------------------------
// 与 pretokenizers::SplitBehavior / SplitPatternKind 一一对应的对外别名,
// 便于不依赖内部头文件即可构造 Split。
enum class SplitBehavior {
    kRemoved,
    kIsolated,
    kMergedWithPrevious,
    kMergedWithNext,
    kContiguous,
};

enum class SplitPatternKind {
    kString,  // 原义字符串(内部转义后作 regex)
    kRegex,   // 直接 regex
};

// behavior 字符串(如 "Isolated")→ SplitBehavior;未知→ kIsolated(HF 默认)
SplitBehavior parse_split_behavior(std::string_view s) noexcept;

// Split 工厂
// pattern:模式字符串;kind=kString 时按原义转义后编译
std::unique_ptr<PreTokenizer> make_split_pretokenizer(std::string pattern, SplitPatternKind kind,
                                                      SplitBehavior behavior, bool invert);

// Sequence 工厂:顺序执行多个 PreTokenizer
std::unique_ptr<PreTokenizer> make_sequence_pretokenizer(std::vector<std::unique_ptr<PreTokenizer>> pretokenizers);

}  // namespace bpe