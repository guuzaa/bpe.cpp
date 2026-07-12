// pretokenizers/sequence.cpp — Sequence PreTokenizer 实现
#include "pretokenizers/sequence.h"
#include <utility>

namespace bpe::pretokenizers {

Sequence::Sequence(std::vector<std::unique_ptr<PreTokenizer>> pretokenizers)
    : pretokenizers_(std::move(pretokenizers)) {
}

void Sequence::pre_tokenize(PreTokenizedString& s) const {
    for (const auto& pt : pretokenizers_) {
        if (pt) {
            pt->pre_tokenize(s);
        }
    }
}

}  // namespace bpe::pretokenizers