// pretokenizers/byte_level.cpp
#include "pretokenizers/byte_level.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "utils/bytes_char.h"
#include "utils/unicode.h"

namespace bpe::pretokenizers {

namespace {

// GPT-2 标准正则(与 HF / openai/gpt-2 一致)
constexpr std::string_view kGpt2Pattern =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

// 判断 codepoint 是否为空白(Unicode White_Space 子集 + byte-char 空格 U+0120)
bool is_whitespace_cp(uint32_t cp) {
    if (cp == util::bytes_to_chars()[0x20]) {
        return true;  // U+0120 'Ġ'
    }
    switch (cp) {
        case 0x09:
        case 0x0A:
        case 0x0B:
        case 0x0C:
        case 0x0D:
        case 0x20:
        case 0x85:
        case 0xA0:
        case 0x1680:
        case 0x2007:
        case 0x2028:
        case 0x2029:
        case 0x202F:
        case 0x205F:
        case 0x3000:
            return true;
        default:
            return cp >= 0x2000 && cp <= 0x200A;
    }
}

// 把 segment 字节映射为 byte-char,并生成相对段起点的 alignment
void map_segment(std::string_view segment, std::string& mapped, std::vector<uint32_t>& align) {
    mapped.clear();
    align.clear();
    mapped.reserve(segment.size() * 2);
    align.reserve(segment.size() * 2);
    for (std::size_t bi = 0; bi < segment.size(); ++bi) {
        const std::size_t before = mapped.size();
        util::append_byte_char(mapped, static_cast<uint8_t>(segment[bi]));
        for (std::size_t k = before; k < mapped.size(); ++k) {
            align.push_back(static_cast<uint32_t>(bi));
        }
    }
}

// 父 text 字节下标 → 相对父 offsets.first 的偏移;空 alignment = 恒等
uint32_t map_parent_byte(uint32_t idx, const std::vector<uint32_t>& parent_alignment) {
    if (parent_alignment.empty()) {
        return idx;
    }
    if (idx >= parent_alignment.size()) {
        return parent_alignment.back();
    }
    return parent_alignment[idx];
}

// text 上 [sp_start, sp_end) 映回「未加 prefix 的 pt.text」字节区间 [rel_start, rel_end)
// did_prefix: text[0] 是我们插入的空格,不计入原文
void span_to_original(std::size_t sp_start, std::size_t sp_end, bool did_prefix, uint32_t& rel_start,
                      uint32_t& rel_end) {
    if (!did_prefix) {
        rel_start = static_cast<uint32_t>(sp_start);
        rel_end = static_cast<uint32_t>(sp_end);
        return;
    }
    // text = ' ' + original; 原文偏移 = max(0, pos - 1)
    const auto shift = [](std::size_t pos) -> uint32_t {
        return pos == 0 ? 0u : static_cast<uint32_t>(pos - 1);
    };
    rel_start = shift(sp_start);
    rel_end = shift(sp_end);
}

}  // namespace

ByteLevel::ByteLevel(bool add_prefix_space, bool trim_offsets, bool use_regex)
    : add_prefix_space_(add_prefix_space), trim_offsets_(trim_offsets), use_regex_(use_regex) {
    if (use_regex_) {
        regex_ = std::make_unique<Regex>(kGpt2Pattern);
    }
}

std::string ByteLevel::encode_bytes(std::string_view s) {
    std::string out;
    out.reserve(s.size() * 2);
    for (char ch : s) {
        util::append_byte_char(out, static_cast<unsigned char>(ch));
    }
    return out;
}

std::string ByteLevel::decode_bytes(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (auto v : util::codepoint_iter(s)) {
        int b = util::char_to_byte(v.cp);
        if (b < 0) {
            out.append(v.bytes.data(), v.bytes.size());
        } else {
            out.push_back(static_cast<char>(b));
        }
    }
    return out;
}

void ByteLevel::pre_tokenize(PreTokenizedString& s) const {
    auto& pretokens = s.tokens();
    std::vector<PreToken> out;
    out.reserve(pretokens.size() * 2);

    for (auto& pt : pretokens) {
        // 是否在本段前插入了 prefix space(与 offset 修正共用同一条件)
        const bool did_prefix = add_prefix_space_ && (pt.text.empty() || pt.text.front() != ' ');

        std::string text;
        if (did_prefix) {
            text.reserve(pt.text.size() + 1);
            text.push_back(' ');
            text.append(pt.text);
        } else {
            text = pt.text;
        }

        std::vector<std::pair<std::size_t, std::size_t>> spans;
        if (use_regex_ && regex_ && regex_->valid()) {
            spans = regex_->find_all(text);
        }
        if (spans.empty() && !text.empty()) {
            spans.emplace_back(0, text.size());
        }

        const uint32_t base = pt.offsets.first;
        for (auto [sp_start, sp_end] : spans) {
            const std::string_view segment(text.data() + sp_start, sp_end - sp_start);

            std::string mapped;
            std::vector<uint32_t> local_align;
            map_segment(segment, mapped, local_align);

            uint32_t rel_start = 0;
            uint32_t rel_end = 0;
            span_to_original(sp_start, sp_end, did_prefix, rel_start, rel_end);

            // offsets / alignment 均相对「本 PreToken 的 offsets.first」:
            //   absolute = offsets.first + alignment[i]
            // 父 alignment 非空时先映到父段内,再减去本段起点,保持上述契约。
            const uint32_t span_origin = map_parent_byte(rel_start, pt.alignment);
            const uint32_t off_start = base + span_origin;
            uint32_t off_end = off_start;
            if (rel_end > rel_start) {
                off_end = base + map_parent_byte(rel_end - 1, pt.alignment) + 1;
                if (off_end < off_start) {
                    off_end = off_start;
                }
            }

            std::vector<uint32_t> align;
            align.reserve(local_align.size());
            for (uint32_t bi : local_align) {
                const uint32_t abs_in_parent = map_parent_byte(rel_start + bi, pt.alignment);
                align.push_back(abs_in_parent - span_origin);
            }

            out.push_back(PreToken{std::move(mapped), {off_start, off_end}, std::move(align)});
        }
    }
    pretokens = std::move(out);
}

std::string ByteLevel::decode(const std::vector<std::string>& tokens) const {
    // 先拼接再 decode,使跨 token 的 multi-byte char 能正确组合
    std::string concat;
    std::size_t total = 0;
    for (const auto& t : tokens) {
        total += t.size();
    }
    concat.reserve(total);
    for (const auto& t : tokens) {
        concat += t;
    }
    return decode_bytes(concat);
}

void ByteLevel::trim_token_offsets(Token& t, std::size_t token_index) const {
    // 对应 HF process_offsets:
    //   按码点统计前导/后导空白;offsets 在 normalized 空间中,对 byte-level
    //   空白(含 Ġ)每码点对应 1 个原始字节,故用码点数加减偏移与 HF 一致。
    // 单次遍历收集码点,同时得到 leading / trailing。
    std::vector<uint32_t> cps;
    cps.reserve(t.value.size());
    for (auto v : util::codepoint_iter(t.value)) {
        cps.push_back(v.cp);
    }
    if (cps.empty()) {
        return;
    }

    std::size_t leading = 0;
    while (leading < cps.size() && is_whitespace_cp(cps[leading])) {
        ++leading;
    }
    std::size_t trailing = 0;
    while (trailing < cps.size() - leading && is_whitespace_cp(cps[cps.size() - 1 - trailing])) {
        ++trailing;
    }
    if (leading == 0 && trailing == 0) {
        return;
    }

    // 仅 token_index==0 视为「序列首 token」(不再用 offsets.first==0 误判)
    if (leading > 0 && token_index == 0 && add_prefix_space_ && leading == 1) {
        leading = 0;
    }

    if (leading > 0) {
        const uint32_t add = static_cast<uint32_t>(leading);
        t.offsets.first = std::min(t.offsets.first + add, t.offsets.second);
    }
    if (trailing > 0) {
        const uint32_t sub = static_cast<uint32_t>(trailing);
        if (t.offsets.second >= t.offsets.first + sub) {
            t.offsets.second -= sub;
        } else {
            t.offsets.second = t.offsets.first;
        }
    }
}

std::vector<Token> ByteLevel::process(std::vector<Token> tokens) const {
    if (!trim_offsets_) {
        return tokens;
    }
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        trim_token_offsets(tokens[i], i);
    }
    return tokens;
}

}  // namespace bpe::pretokenizers
