// pretokenizers/split.h — Split PreTokenizer
//
// 对照 HuggingFace tokenizers 的 Split pre-tokenizer:
//   - 用 pattern(regex / literal string)在 text 上找出匹配区间
//   - 按 behavior 决定匹配区间与夹在中间的「空隙」如何产出 PreToken
//   - invert=true 时对调「match / gap」语义,实现「保留匹配、移除空隙」等
//
// 行为枚举与 HF SplitDelimiterBehavior 一一对应(见 normalizer.rs 注释):
//   以分隔符 '-' 切分 "the-final--countdown" 为例
//     kRemoved           => ["the", "final", "countdown"]
//     kIsolated          => ["the", "-", "final", "-", "-", "countdown"]
//     kMergedWithPrevious=> ["the-", "final-", "-", "countdown"]
//     kMergedWithNext    => ["the", "-final", "-", "-countdown"]
//     kContiguous        => ["the", "-", "final", "--", "countdown"]
//
// 注:此处 Split 仅按字节区间切分,不做 byte-char 映射,alignment 维持恒等。
// 与 ByteLevel 组合时,Split 通常位于 ByteLevel 之前(在原文/Unicode 空间切分)。
//
// SplitBehavior / SplitPatternKind 定义在公共头 bpe/pretokenizers.h 中,
// 这里在 bpe::pretokenizers:: 中以 using 别名引出,避免重复定义导致歧义。
#pragma once

#include <memory>
#include <string>

#include "bpe/interfaces.h"
#include "bpe/pretokenizers.h"
#include "pretokenizers/regex_engine.h"

namespace bpe::pretokenizers {

using SplitBehavior = ::bpe::SplitBehavior;
using SplitPatternKind = ::bpe::SplitPatternKind;

class Split : public PreTokenizer {
   public:
    // pattern:模式字符串;kind=kString 时按原义转义后编译
    // behavior:匹配/空隙如何产出 PreToken
    // invert:true 时反转 match/gap 角色(对调语义)
    Split(std::string pattern, SplitPatternKind kind, SplitBehavior behavior, bool invert);

    void pre_tokenize(PreTokenizedString& s) const override;

    const std::string& pattern() const noexcept {
        return pattern_;
    }

    SplitPatternKind pattern_kind() const noexcept {
        return pattern_kind_;
    }

    SplitBehavior behavior() const noexcept {
        return behavior_;
    }

    bool invert() const noexcept {
        return invert_;
    }

    bool valid() const noexcept {
        return regex_ && regex_->valid();
    }

   private:
    std::string pattern_;
    SplitPatternKind pattern_kind_;
    SplitBehavior behavior_;
    bool invert_;
    std::unique_ptr<Regex> regex_;
};

}  // namespace bpe::pretokenizers