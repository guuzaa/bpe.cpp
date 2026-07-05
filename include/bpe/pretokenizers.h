// bpe/pretokenizers.h — PreTokenizer 工厂与对外枚举
#pragma once

#include <memory>

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

}  // namespace bpe
