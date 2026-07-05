// utils/bytes_char.cpp
#include "utils/bytes_char.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace bpe::util {

namespace {

// 构造与 HF byte_level.rs::bytes_char 完全一致的映射表
std::array<uint32_t, 256> build_table() {
    std::array<uint32_t, 256> cs{};
    // 第一段:可打印 ASCII + 拉丁 1 段
    std::vector<uint8_t> bs;
    bs.reserve(256);
    for (int b = 0x21; b <= 0x7E; ++b) {
        bs.push_back(static_cast<uint8_t>(b));
    }
    for (int b = 0xA1; b <= 0xAC; ++b) {
        bs.push_back(static_cast<uint8_t>(b));
    }
    for (int b = 0xAE; b <= 0xFF; ++b) {
        bs.push_back(static_cast<uint8_t>(b));
    }

    // cs[i] = bs[i] 自身(对前 223 个)
    for (std::size_t i = 0; i < bs.size(); ++i) {
        cs[bs[i]] = bs[i];
    }

    // 其余 33 个控制字节 → U+0100 + n
    uint32_t n = 0;
    for (int b = 0; b <= 0xFF; ++b) {
        // 若 b 不在 bs 中
        bool in_bs = false;
        for (auto x : bs) {
            if (x == b) {
                in_bs = true;
                break;
            }
        }
        if (!in_bs) {
            cs[static_cast<uint8_t>(b)] = 0x100u + n;
            ++n;
        }
    }
    return cs;
}

}  // namespace

const std::array<uint32_t, 256>& bytes_to_chars() noexcept {
    static const std::array<uint32_t, 256> table = build_table();
    return table;
}

int char_to_byte(uint32_t cp) noexcept {
    // 反向查表:由于 code point 分布稀疏,用一次性构建的 unordered_map
    static const std::unordered_map<uint32_t, uint8_t> rev = [] {
        std::unordered_map<uint32_t, uint8_t> m;
        const auto& fwd = bytes_to_chars();
        for (int b = 0; b <= 0xFF; ++b) {
            m.emplace(fwd[static_cast<uint8_t>(b)], static_cast<uint8_t>(b));
        }
        return m;
    }();
    auto it = rev.find(cp);
    return it == rev.end() ? -1 : it->second;
}

std::string byte_to_string(uint8_t b) {
    uint32_t cp = bytes_to_chars()[b];
    std::string out;
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
        // U+10000 以上(本表不会出现)
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

}  // namespace bpe::util
