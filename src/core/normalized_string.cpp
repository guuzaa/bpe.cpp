// core/normalized_string.cpp — NormalizedString 偏移对齐实现
#include <cstdint>
#include <cstring>

#include "bpe/interfaces.h"

namespace bpe {

std::pair<uint32_t, uint32_t> NormalizedString::align_to_original(uint32_t norm_start, uint32_t norm_end) const {
    if (aligned_to_original_.empty()) {
        return {norm_start, norm_end};
    }
    auto clamp = [&](uint32_t i) {
        if (i >= aligned_to_original_.size()) {
            return aligned_to_original_.empty() ? i : aligned_to_original_.back();
        }
        return aligned_to_original_[i];
    };
    uint32_t s = clamp(norm_start);
    // 结束偏移取 norm_end - 1 对应的 original + 1;若 norm_end == 0,则 s==0
    uint32_t e;
    if (norm_end == 0 || norm_end > aligned_to_original_.size()) {
        if (aligned_to_original_.empty()) {
            e = norm_end;
        } else {
            e = aligned_to_original_.back() + 1;
        }
    } else {
        e = aligned_to_original_[norm_end - 1] + 1;
    }
    if (e < s) {
        e = s;
    }
    return {s, e};
}

void NormalizedString::replace_range(uint32_t norm_start, uint32_t norm_end, std::string_view replacement) {
    // 简化实现:支持"等长替换"和"删除"两种最常用形式
    // 替换 [start, end) 为 replacement,新 normalized 的对齐表:
    //   保留 [0,start),插入 replacement(每个字节对应 start 的 original),
    //   保留原 [end, size)
    const std::size_t start = norm_start;
    const std::size_t end = norm_end;
    if (start > normalized_.size()) {
        return;
    }
    if (end > normalized_.size()) {
        return;
    }
    if (start > end) {
        return;
    }

    std::string new_norm;
    std::vector<uint32_t> new_align;
    new_norm.reserve(normalized_.size() - (end - start) + replacement.size());
    new_align.reserve(new_norm.capacity());

    // 前 [0, start)
    new_norm.append(normalized_.data(), start);
    new_align.insert(new_align.end(), aligned_to_original_.begin(),
                     aligned_to_original_.begin() + static_cast<std::ptrdiff_t>(start));

    // replacement 每个字节对应原 start 位置的 original 偏移
    uint32_t rep_orig =
        (start < aligned_to_original_.size())
            ? aligned_to_original_[start]
            : (aligned_to_original_.empty() ? static_cast<uint32_t>(start) : aligned_to_original_.back());
    for (std::size_t i = 0; i < replacement.size(); ++i) {
        new_norm.push_back(replacement[i]);
        new_align.push_back(rep_orig);
    }

    // 后 [end, size)
    new_norm.append(normalized_.data() + end, normalized_.size() - end);
    new_align.insert(new_align.end(), aligned_to_original_.begin() + static_cast<std::ptrdiff_t>(end),
                     aligned_to_original_.end());
    normalized_ = std::move(new_norm);
    aligned_to_original_ = std::move(new_align);
}

void NormalizedString::append(std::string_view s) {
    uint32_t orig =
        aligned_to_original_.empty() ? static_cast<uint32_t>(normalized_.size()) : aligned_to_original_.back() + 1;
    for (char c : s) {
        normalized_.push_back(c);
        aligned_to_original_.push_back(orig);
    }
}

void NormalizedString::lowercase() {
    // ASCII + Latin-1 tolower;不依赖 ICU
    for (char& c : normalized_) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc >= 'A' && uc <= 'Z') {
            c = static_cast<char>(uc + ('a' - 'A'));
        } else if (uc >= 0xC0 && uc <= 0xDE && uc != 0xD7) {
            // 拉丁 1 大写段 → 小写段 +0x20(0xD7 乘号除外)
            c = static_cast<char>(uc + 0x20);
        }
    }
    // 对齐表不变:长度未变
}

void NormalizedString::uppercase() {
    for (char& c : normalized_) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc >= 'a' && uc <= 'z') {
            c = static_cast<char>(uc - ('a' - 'A'));
        } else if (uc >= 0xE0 && uc <= 0xFE && uc != 0xF7) {
            c = static_cast<char>(uc - 0x20);
        }
    }
}

void NormalizedString::strip() {
    // 找首个非空白
    std::size_t start = 0;
    while (start < normalized_.size()) {
        unsigned char c = static_cast<unsigned char>(normalized_[start]);
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f') {
            ++start;
        } else {
            break;
        }
    }
    std::size_t end = normalized_.size();
    while (end > start) {
        unsigned char c = static_cast<unsigned char>(normalized_[end - 1]);
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f') {
            --end;
        } else {
            break;
        }
    }
    if (start == 0 && end == normalized_.size()) {
        return;
    }
    replace_range(0, static_cast<uint32_t>(start), "");
    replace_range(static_cast<uint32_t>(end - start), static_cast<uint32_t>(normalized_.size()), "");
}

}  // namespace bpe