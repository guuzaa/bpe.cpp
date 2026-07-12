// utils/bytes_char.cpp
#include "utils/bytes_char.h"
#include <cstdint>
#include <string>
#include <unordered_map>

namespace bpe::util {

namespace {

// 构造与 HF byte_level.rs::bytes_char 完全一致的映射表
constexpr std::array<uint32_t, 256> build_table() {
    std::array<uint32_t, 256> cs{};
    // 可见字节(可打印 ASCII + 拉丁 1 段)→ 原字节自身
    for (uint32_t b = 0x21; b <= 0x7Eu; ++b) {
        cs[b] = b;
    }
    for (uint32_t b = 0xA1; b <= 0xACu; ++b) {
        cs[b] = b;
    }
    for (uint32_t b = 0xAE; b <= 0xFFu; ++b) {
        cs[b] = b;
    }

    // 其余控制字节 → U+0100 + n(按字节升序连续编号)
    // 可见字节均 ≥ 0x21,故 cs[b]==0 即可作为"控制字节"判据
    uint32_t n = 0;
    for (uint32_t b = 0; b <= 0xFFu; ++b) {
        if (cs[b] == 0) {
            cs[b] = 0x100u + n;
            ++n;
        }
    }
    return cs;
}

}  // namespace

const std::array<uint32_t, 256>& bytes_to_chars() noexcept {
    static constexpr std::array<uint32_t, 256> table = build_table();
    return table;
}

int char_to_byte(uint32_t cp) noexcept {
    // 反向查表:由于 code point 分布稀疏,用一次性构建的 unordered_map
    static const std::unordered_map<uint32_t, uint8_t> rev = [] {
        std::unordered_map<uint32_t, uint8_t> m;
        const auto& fwd = bytes_to_chars();
        for (uint32_t b = 0; b <= 0xFFu; ++b) {
            m.emplace(fwd[b], static_cast<uint8_t>(b));
        }
        return m;
    }();
    auto it = rev.find(cp);
    return it == rev.end() ? -1 : it->second;
}

void append_byte_char(std::string& out, uint8_t b) {
    const uint32_t cp = bytes_to_chars()[b];
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::string byte_to_string(uint8_t b) {
    std::string out;
    out.reserve(3);
    append_byte_char(out, b);
    return out;
}

}  // namespace bpe::util
