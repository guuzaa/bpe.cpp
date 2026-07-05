// src/pretokenizers/factories.cpp — PreTokenizer 工厂实现
#include "bpe/pretokenizers.h"
#include "pretokenizers/byte_level.h"

namespace bpe {

std::unique_ptr<PreTokenizer> make_byte_level_pretokenizer(bool add_prefix_space, bool trim_offsets, bool use_regex) {
    return std::make_unique<pretokenizers::ByteLevel>(add_prefix_space, trim_offsets, use_regex);
}

}  // namespace bpe
