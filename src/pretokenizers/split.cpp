// pretokenizers/split.cpp — Split PreTokenizer 实现
//
// 算法对照 HF tokenizers `normalizer.rs SplitDelimiterBehavior` 与
// `pattern.rs find_matches`:
//   1. regex.find_all(text) → 得到非空匹配区间集
//   2. 用匹配与匹配间空隙构造覆盖整段 text 的 segments[(start,end,is_match)]
//      (空隙/匹配长度均为 0 时跳过,与 HF 一致)
//   3. invert=true 时对调 is_match
//   4. 按 behavior 合并/过滤 segments,产出最终区间
//
// PreToken 的 alignment 保持恒等(Split 不改变 text 长度);若父 PreToken
// 有 alignment(例如 Split 位于 ByteLevel 之后),按父 alignment 透传。
#include "pretokenizers/split.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace bpe::pretokenizers {

namespace {

// 对原义字符串做 PCRE2 转义,对照 Rust `regex` crate 的 escape 实现
std::string escape_regex(std::string_view s) {
    std::string out;
    out.reserve(s.size() * 2);
    for (char ch : s) {
        switch (ch) {
            case '\\':
            case '.':
            case '+':
            case '*':
            case '?':
            case '(':
            case ')':
            case '|':
            case '[':
            case ']':
            case '{':
            case '}':
            case '^':
            case '$':
                out.push_back('\\');
                out.push_back(ch);
                break;
            default:
                out.push_back(ch);
        }
    }
    return out;
}

// 父 text 字节下标 → 相对父 offsets.first 的偏移;空 alignment = 恒等
// (与 byte_level.cpp 中同名 helper 等价,放在本文件内以保持独立)
uint32_t map_parent_byte(uint32_t idx, const std::vector<uint32_t>& parent_alignment) {
    if (parent_alignment.empty()) {
        return idx;
    }
    if (idx >= parent_alignment.size()) {
        return parent_alignment.back();
    }
    return parent_alignment[idx];
}

struct Segment {
    std::size_t start;
    std::size_t end;
    bool is_match;  // regex match=true,gap=false(invert 之后再统一翻转)
};

// 把 regex 匹配结果转换为覆盖整段 text 的 segments(空段已剔除)
std::vector<Segment> build_segments(std::string_view text, const Regex& regex) {
    std::vector<Segment> segs;
    auto matches = regex.find_all(text);
    segs.reserve(matches.size() * 2 + 1);
    std::size_t prev = 0;
    for (const auto& [ms, me] : matches) {
        if (prev < ms) {
            segs.push_back({prev, ms, false});
        }
        segs.push_back({ms, me, true});
        prev = me;
    }
    if (prev < text.size()) {
        segs.push_back({prev, text.size(), false});
    }
    return segs;
}

// 应用 behavior,生成最终保留的区间列表(已剔除空段)
std::vector<std::pair<std::size_t, std::size_t>> apply_behavior(const std::vector<Segment>& segs,
                                                                SplitBehavior behavior) {
    std::vector<std::pair<std::size_t, std::size_t>> out;
    auto push_if_nonempty = [&](std::size_t s, std::size_t e) {
        if (s < e) {
            out.emplace_back(s, e);
        }
    };

    switch (behavior) {
        case SplitBehavior::kIsolated:
            for (const auto& s : segs) {
                push_if_nonempty(s.start, s.end);
            }
            break;

        case SplitBehavior::kRemoved:
            for (const auto& s : segs) {
                if (!s.is_match) {
                    push_if_nonempty(s.start, s.end);
                }
            }
            break;

        case SplitBehavior::kContiguous: {
            bool previous_match = false;
            bool first = true;
            for (const auto& s : segs) {
                if (!first && s.is_match == previous_match) {
                    if (!out.empty()) {
                        out.back().second = s.end;
                    } else {
                        push_if_nonempty(s.start, s.end);
                    }
                } else {
                    push_if_nonempty(s.start, s.end);
                }
                previous_match = s.is_match;
                first = false;
            }
            break;
        }

        case SplitBehavior::kMergedWithPrevious: {
            bool previous_match = false;
            for (const auto& s : segs) {
                if (s.is_match && !previous_match) {
                    if (!out.empty()) {
                        out.back().second = s.end;
                    } else {
                        push_if_nonempty(s.start, s.end);
                    }
                } else {
                    push_if_nonempty(s.start, s.end);
                }
                previous_match = s.is_match;
            }
            break;
        }

        case SplitBehavior::kMergedWithNext: {
            bool previous_match = false;
            for (auto it = segs.rbegin(); it != segs.rend(); ++it) {
                const auto& s = *it;
                if (s.is_match && !previous_match) {
                    if (!out.empty()) {
                        out.back().first = s.start;
                    } else {
                        push_if_nonempty(s.start, s.end);
                    }
                } else {
                    push_if_nonempty(s.start, s.end);
                }
                previous_match = s.is_match;
            }
            std::reverse(out.begin(), out.end());
            break;
        }
    }
    return out;
}

}  // namespace

Split::Split(std::string pattern, SplitPatternKind kind, SplitBehavior behavior, bool invert)
    : pattern_(std::move(pattern)), pattern_kind_(kind), behavior_(behavior), invert_(invert) {
    std::string compiled = (kind == SplitPatternKind::kString) ? escape_regex(pattern_) : pattern_;
    regex_ = std::make_unique<Regex>(compiled);
}

void Split::pre_tokenize(PreTokenizedString& s) const {
    auto& pretokens = s.tokens();
    std::vector<PreToken> out;
    out.reserve(pretokens.size() * 2);

    for (const auto& pt : pretokens) {
        // 空 PreToken 直接透传(保持下游行为一致)
        if (pt.text.empty()) {
            out.push_back(pt);
            continue;
        }

        std::vector<Segment> segs;
        if (regex_ && regex_->valid()) {
            segs = build_segments(pt.text, *regex_);
        }
        // 无 regex / 无匹配 → 整段视作 gap
        if (segs.empty()) {
            segs.push_back({0, pt.text.size(), false});
        }

        if (invert_) {
            for (auto& seg : segs) {
                seg.is_match = !seg.is_match;
            }
        }

        auto final_segs = apply_behavior(segs, behavior_);
        const uint32_t base = pt.offsets.first;
        for (auto& [start, end] : final_segs) {
            const uint32_t span_origin = map_parent_byte(static_cast<uint32_t>(start), pt.alignment);
            const uint32_t off_start = base + span_origin;
            uint32_t off_end = off_start;
            if (end > start) {
                off_end = base + map_parent_byte(static_cast<uint32_t>(end - 1), pt.alignment) + 1;
                if (off_end < off_start) {
                    off_end = off_start;
                }
            }

            std::vector<uint32_t> align;
            if (!pt.alignment.empty()) {
                align.reserve(end - start);
                for (std::size_t i = start; i < end; ++i) {
                    const uint32_t abs_in_parent = map_parent_byte(static_cast<uint32_t>(i), pt.alignment);
                    align.push_back(abs_in_parent - span_origin);
                }
            }

            out.push_back(
                PreToken{std::string(pt.text.data() + start, end - start), {off_start, off_end}, std::move(align)});
        }
    }
    pretokens = std::move(out);
}

}  // namespace bpe::pretokenizers