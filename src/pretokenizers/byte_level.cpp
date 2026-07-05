// pretokenizers/byte_level.cpp
#include "pretokenizers/byte_level.h"
#include <cstdint>
#include <cstring>
#include <vector>

#include "utils/bytes_char.h"
#include "utils/unicode.h"

namespace bpe::pretokenizers {

namespace {

// GPT-2 标准正则(与 HF / openai/gpt-2 一致)
constexpr std::string_view kGpt2Pattern =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

// 判断 codepoint 是否为空白(对应 Rust 的 char::is_whitespace)
// 取 Unicode White_Space 属性的常见子集 + byte-char 的空格(U+0120)
bool is_whitespace_cp(uint32_t cp) {
    // 标准 Unicode 空白
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
        // 0x2000..0x200A, 0x202F, 0x205F, 0x3000
        case 0x202F:
        case 0x205F:
        case 0x3000:
            return true;
        default:
            if (cp >= 0x2000 && cp <= 0x200A) {
                return true;
            }
            return false;
    }
}

}  // namespace

ByteLevel::ByteLevel(bool add_prefix_space, bool trim_offsets, bool use_regex)
    : add_prefix_space_(add_prefix_space), trim_offsets_(trim_offsets), use_regex_(use_regex), regex_(kGpt2Pattern) {
}

std::string ByteLevel::encode_bytes(std::string_view s) {
    // 把每个字节映射为对应的 byte-char(可能 1~3 字节 UTF-8)
    const auto& table = util::bytes_to_chars();
    std::string out;
    out.reserve(s.size() * 2);  // 平均估计
    for (char ch : s) {
        out += util::byte_to_string(static_cast<unsigned char>(ch));
    }
    (void)table;
    return out;
}

std::string ByteLevel::decode_bytes(std::string_view s) {
    // 按 UTF-8 码点遍历,每个码点查反向表得到字节
    std::string out;
    out.reserve(s.size());
    util::codepoint_iter it(s);
    for (auto v : it) {
        int b = util::char_to_byte(v.cp);
        if (b < 0) {
            // 非映射字符:保留原 UTF-8 字节(对应 HF unwrap_or_else 分支)
            out.append(v.bytes.data(), v.bytes.size());
        } else {
            out.push_back(static_cast<char>(b));
        }
    }
    return out;
}

void ByteLevel::pre_tokenize(PreTokenizedString& s) const {
    auto& pretokens = s.tokens();
    // 准备输出
    std::vector<PreToken> out;
    out.reserve(pretokens.size() * 2);

    for (auto& pt : pretokens) {
        // 1. 取出 normalized 子串(此处用 pt.text;M2 的 NormalizedString 暂未完整对齐)
        std::string text = pt.text;

        // 2. add_prefix_space:若不以 ' ' 开头则前置一个空格
        if (add_prefix_space_ && (text.empty() || text.front() != ' ')) {
            text.insert(text.begin(), ' ');
        }

        // 3. 用正则切分(若 use_regex=false,整段作为一个 split)
        std::vector<std::pair<std::size_t, std::size_t>> spans;
        if (use_regex_ && regex_.valid()) {
            spans = regex_.find_all(text);
        }
        if (spans.empty()) {
            // 无切分(或 use_regex=false):整段作为单个 split
            if (!text.empty()) {
                spans.emplace_back(0, text.size());
            }
        }

        // 4. 对每个 split 做字节映射
        // 偏移对齐:add_prefix_space 改变了 text 与原 pt.offsets 的关系
        // 这里我们直接以映射后的 byte-char 字符串作为 PreToken.text,
        // 偏移以原 pt.offsets 起点为基准 + (split.start, split.end) 在 text 中的位置
        // 注意:add_prefix_space 若加了 1 字节空格,需要把 split 偏移扣 1(对齐回原文)
        const uint32_t base = pt.offsets.first;
        for (auto [sp_start, sp_end] : spans) {
            std::string segment = text.substr(sp_start, sp_end - sp_start);

            // 逐字节映射为 byte-char,同时构建 alignment 表:
            // alignment[i] = mapped 第 i 字节对应 segment 内的第几个原始字节
            std::string mapped;
            std::vector<uint32_t> align;
            mapped.reserve(segment.size() * 2);
            align.reserve(segment.size() * 2);
            for (std::size_t bi = 0; bi < segment.size(); ++bi) {
                std::string bc = util::byte_to_string(static_cast<uint8_t>(segment[bi]));
                for (std::size_t k = 0; k < bc.size(); ++k) {
                    mapped.push_back(bc[k]);
                    align.push_back(static_cast<uint32_t>(bi));
                }
            }

            // 计算对齐到原文的字节偏移:
            //   - 若 add_prefix_space 且加了空格,则原文偏移 = base + sp_start - 1
            //     (但 sp_start=0 时,这个 split 包含了 prefix space,偏移应 clamp 到 base)
            //   - 否则 = base + sp_start
            uint32_t off_start = base;
            uint32_t off_end = base;
            if (add_prefix_space_ && (pt.text.empty() || pt.text.front() != ' ')) {
                // 我们在 text 前加了一个空格,所以 split 的实际原文偏移要 -1
                // 但若 sp_start == 0(split 含 prefix space),原文偏移起点 = base
                int32_t shift_start = static_cast<int32_t>(sp_start) - 1;
                int32_t shift_end = static_cast<int32_t>(sp_end) - 1;
                if (shift_start < 0) {
                    shift_start = 0;
                }
                if (shift_end < 0) {
                    shift_end = 0;
                }
                off_start = base + static_cast<uint32_t>(shift_start);
                off_end = base + static_cast<uint32_t>(shift_end);
            } else {
                off_start = base + static_cast<uint32_t>(sp_start);
                off_end = base + static_cast<uint32_t>(sp_end);
            }

            out.push_back(PreToken{std::move(mapped), {off_start, off_end}, std::move(align)});
        }
    }
    pretokens = std::move(out);
}

std::string ByteLevel::decode(const std::vector<std::string>& tokens) const {
    // 把所有 token 字符串拼起来,再一次性 decode_bytes
    // 这样单个 token 末尾的 byte-char 与下一 token 开头的 byte-char 能正确组合
    std::string concat;
    for (const auto& t : tokens) {
        concat += t;
    }
    return decode_bytes(concat);
}

void ByteLevel::trim_token_offsets(Token& t, std::size_t token_index) const {
    // 对应 HF process_offsets:
    //   - 计算 t.value 中的前导/后导空白(用 byte-char 表示,空格 = U+0120 'Ġ')
    //   - 前导:若是首个 token 且 add_prefix_space 且只有 1 个前导空格 → 保留(我们加的)
    //   - 否则前导偏移 += leading
    //   - 后导偏移 -= trailing
    const uint32_t space_cp = util::bytes_to_chars()[0x20];  // U+0120

    // 统计 t.value 的前导/后导"空白"码点数
    // 注意:t.value 是 byte-char 字符串,空格对应 'Ġ'(U+0120, 2 字节)
    std::size_t leading = 0, trailing = 0;
    {
        // 前导
        util::codepoint_iter it(t.value);
        for (auto v : it) {
            if (v.cp == space_cp || is_whitespace_cp(v.cp)) {
                ++leading;
            } else {
                break;
            }
        }
        // 后导:反向扫描(用解码后的码点 vector)
        std::vector<uint32_t> cps;
        for (auto v : util::codepoint_iter(t.value)) {
            cps.push_back(v.cp);
        }
        for (auto rit = cps.rbegin(); rit != cps.rend(); ++rit) {
            if (*rit == space_cp || is_whitespace_cp(*rit)) {
                ++trailing;
            } else {
                break;
            }
        }
    }

    if (leading == 0 && trailing == 0) {
        return;
    }

    if (leading > 0) {
        bool is_first = (token_index == 0) || (t.offsets.first == 0);
        if (is_first && add_prefix_space_ && leading == 1) {
            leading = 0;  // 我们加的那个前导空格,保留
        }
        if (leading > 0) {
            uint32_t add = static_cast<uint32_t>(leading);
            uint32_t new_start =
                (t.offsets.first + add < t.offsets.second) ? (t.offsets.first + add) : t.offsets.second;
            t.offsets.first = new_start;
        }
    }
    if (trailing > 0 && t.offsets.second >= trailing) {
        uint32_t sub = static_cast<uint32_t>(trailing);
        uint32_t new_end = (t.offsets.second - sub > t.offsets.first) ? (t.offsets.second - sub) : t.offsets.first;
        t.offsets.second = new_end;
    }
}

std::vector<Token> ByteLevel::process(std::vector<Token> tokens) const {
    if (!trim_offsets_) {
        return tokens;
    }
    const std::size_t n = tokens.size();
    for (std::size_t i = 0; i < n; ++i) {
        trim_token_offsets(tokens[i], i);
    }
    return tokens;
}

}  // namespace bpe::pretokenizers
