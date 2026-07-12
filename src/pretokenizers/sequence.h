// pretokenizers/sequence.h — Sequence PreTokenizer
//
// 顺序执行多个 PreTokenizer,把前一个的输出当作后一个的输入。
// 与 HF `Sequence` 行为一致:每一步的 PreTokenizedString 在原地被改写,
// 下一个子 PreTokenizer 在已切分后的 PreToken 列表上继续工作。
//
// 典型用法:DeepSeek v3.2 等模型把若干 `Split` 放在 `ByteLevel` 之前,
// 形成 Sequence = [Split(digits), Split(CJK), Split(main-regex), ByteLevel]。
#pragma once

#include <memory>
#include <vector>

#include "bpe/interfaces.h"

namespace bpe::pretokenizers {

class Sequence : public PreTokenizer {
   public:
    explicit Sequence(std::vector<std::unique_ptr<PreTokenizer>> pretokenizers);

    void pre_tokenize(PreTokenizedString& s) const override;

    const std::vector<std::unique_ptr<PreTokenizer>>& pretokenizers() const noexcept {
        return pretokenizers_;
    }

   private:
    std::vector<std::unique_ptr<PreTokenizer>> pretokenizers_;
};

}  // namespace bpe::pretokenizers