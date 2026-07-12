// src/decoders/factories.cpp — Decoder 工厂实现
#include "bpe/decoders.h"
#include "pretokenizers/byte_level.h"

namespace bpe {

std::unique_ptr<Decoder> make_byte_level_decoder() {
    // Decoder 路径不需要正则 / trim / prefix
    return std::make_unique<pretokenizers::ByteLevel>(/*add_prefix_space=*/false, /*trim_offsets=*/false,
                                                      /*use_regex=*/false);
}

}  // namespace bpe
