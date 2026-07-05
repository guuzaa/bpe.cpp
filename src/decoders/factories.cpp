// src/decoders/factories.cpp — Decoder 工厂实现
#include "bpe/decoders.h"
#include "pretokenizers/byte_level.h"

namespace bpe {

std::unique_ptr<Decoder> make_byte_level_decoder() {
    // ByteLevel 作为 Decoder 时,trim_offsets / use_regex / add_prefix_space 均无关
    // 使用默认构造(add_prefix_space=true),decode 只用 decode_bytes 路径
    return std::make_unique<pretokenizers::ByteLevel>(true, true, true);
}

}  // namespace bpe
