// bpe/decoders.h — Decoder 工厂
#pragma once

#include <memory>

#include "bpe/interfaces.h"

namespace bpe {

// ByteLevel decoder:byte-char → 字节 → UTF-8
std::unique_ptr<Decoder> make_byte_level_decoder();

}  // namespace bpe
