// utils/unicode.h — UTF-8 工具(M1 阶段仅需按码点迭代 + 字节长度)
//
// 后续阶段会扩展为 unicode_properties (L/N/P/White 类别);M1 不依赖。
#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string_view>
#include <utility>

namespace bpe::util {

// Unicode 替换字符 U+FFFD,用于非法/无法解码的 UTF-8 序列
constexpr uint32_t kReplacementChar = 0xFFFD;

// 判断首字节给出的 UTF-8 序列长度;非法首字节返回 1(单字节替换)
inline int utf8_seq_len(unsigned char c) {
    if (c < 0x80) {
        return 1;
    }
    if ((c >> 5) == 0b110) {
        return 2;
    }
    if ((c >> 4) == 0b1110) {
        return 3;
    }
    if ((c >> 3) == 0b11110) {
        return 4;
    }
    return 1;  // 非法首字节,按单字节处理
}

// 解码一个 UTF-8 序列,返回 (codepoint, 字节长度);非法序列返回 (U+FFFD, 1)
// `p` 指向首字节,`end` 指向缓冲区末尾
inline std::pair<uint32_t, int> decode_utf8(const char* p, const char* end) {
    if (p >= end) {
        return {0, 0};
    }
    auto c0 = static_cast<unsigned char>(*p);

    auto get = [&](int i) { return static_cast<unsigned char>(p[i]); };
    auto check_cont = [](unsigned char b) { return (b >> 6) == 0b10; };

    int len = utf8_seq_len(c0);
    if (p + len > end) {
        return {kReplacementChar, 1};
    }

    if (len == 1) {
        if (c0 < 0x80) {
            return {c0, 1};
        }
        return {kReplacementChar, 1};  // 非法 continuation/孤立高位字节
    }

    if (len == 2) {
        auto c1 = get(1);
        if (!check_cont(c1)) {
            return {kReplacementChar, 1};
        }
        uint32_t cp = static_cast<uint32_t>((c0 & 0x1F) << 6 | (c1 & 0x3F));
        if (cp < 0x80) {
            return {kReplacementChar, 1};  // over-long
        }
        return {cp, 2};
    }
    if (len == 3) {
        auto c1 = get(1), c2 = get(2);
        if (!check_cont(c1) || !check_cont(c2)) {
            return {kReplacementChar, 1};
        }
        uint32_t cp = static_cast<uint32_t>((c0 & 0x0F) << 12 | ((c1 & 0x3F) << 6) | (c2 & 0x3F));
        if (cp < 0x800) {
            return {kReplacementChar, 1};
        }
        if (cp >= 0xD800 && cp <= 0xDFFF) {
            return {kReplacementChar, 1};  // surrogate
        }
        return {cp, 3};
    }
    // len == 4
    auto c1 = get(1), c2 = get(2), c3 = get(3);
    if (!check_cont(c1) || !check_cont(c2) || !check_cont(c3)) {
        return {kReplacementChar, 1};
    }
    uint32_t cp = static_cast<uint32_t>((c0 & 0x07) << 18 | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F));
    if (cp < 0x10000 || cp > 0x10FFFF) {
        return {kReplacementChar, 1};
    }
    return {cp, 4};
}

// 按 UTF-8 码点遍历字符串,每步返回 (codepoint, 字节长度)
class codepoint_iter {
   public:
    codepoint_iter(std::string_view s) : s_(s) {
    }

    struct value {
        uint32_t cp;
        int len;
        std::string_view bytes;  // 当前码点的字节切片
    };

    class iterator {
       public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = value;
        using difference_type = std::ptrdiff_t;
        using pointer = const value*;
        using reference = const value&;

        iterator(std::string_view s, bool end_flag) : s_(s), end_(end_flag) {
            if (!end_) {
                advance();
            }
        }

        reference operator*() const {
            return v_;
        }

        pointer operator->() const {
            return &v_;
        }

        iterator& operator++() {
            if (!end_) {
                advance();
            }
            return *this;
        }

        iterator operator++(int) {
            auto t = *this;
            ++(*this);
            return t;
        }

        bool operator==(const iterator& o) const noexcept {
            if (end_ && o.end_) {
                return true;
            }
            if (end_ != o.end_) {
                return false;
            }
            return v_.bytes.data() == o.v_.bytes.data();
        }

        bool operator!=(const iterator& o) const noexcept {
            return !(*this == o);
        }

       private:
        void advance() {
            if (pos_ == s_.size()) {
                end_ = true;
                return;
            }
            auto [cp, len] = decode_utf8(s_.data() + pos_, s_.data() + s_.size());
            v_ = value{cp, len, s_.substr(pos_, static_cast<std::size_t>(len))};
            pos_ += static_cast<std::size_t>(len);
        }

        std::string_view s_;
        std::size_t pos_ = 0;
        bool end_;
        value v_{};
    };

    iterator begin() const {
        return iterator(s_, false);
    }

    iterator end() const {
        return iterator(s_, true);
    }

   private:
    std::string_view s_;
};

}  // namespace bpe::util
