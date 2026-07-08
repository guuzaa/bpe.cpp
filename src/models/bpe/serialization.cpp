// models/bpe/serialization.cpp
#include "models/bpe/serialization.h"
#include <fstream>
#include <utility>

#include "absl/strings/str_split.h"
#include "nlohmann/json.hpp"

namespace bpe::bpe_internal {

absl::StatusOr<Vocab> read_vocab_json(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        return absl::NotFoundError("cannot open vocab.json: " + path);
    }
    nlohmann::json j;
    try {
        ifs >> j;
    } catch (const std::exception& e) {
        return absl::InvalidArgumentError("failed to parse vocab.json: " + std::string(e.what()));
    }
    if (!j.is_object()) {
        return absl::InvalidArgumentError("vocab.json must be a JSON object");
    }
    Vocab vocab;
    vocab.reserve(j.size());
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (!it.value().is_number_integer()) {
            return absl::InvalidArgumentError("vocab.json value for key '" + it.key() + "' is not an integer");
        }
        vocab.emplace(it.key(), static_cast<TokenId>(it.value().get<int64_t>()));
    }
    return vocab;
}

absl::StatusOr<MergeList> read_merges_txt(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        return absl::NotFoundError("cannot open merges.txt: " + path);
    }
    MergeList merges;
    std::string line;
    while (std::getline(ifs, line)) {
        // 跳过空行与 "#version" 之类注释行(同 HF 行为)
        if (line.empty() || line[0] == '#') {
            continue;
        }
        // 按第一个空格拆分 "a b"(MaxSplits 保留 b 中后续空格)
        std::vector<std::string> parts = absl::StrSplit(line, absl::MaxSplits(' ', 1));
        if (parts.size() != 2) {
            continue;  // 格式不合法,跳过
        }
        // 去掉 b 末尾可能的 \r(Windows 换行)
        std::string& b = parts[1];
        if (!b.empty() && b.back() == '\r') {
            b.pop_back();
        }
        merges.emplace_back(std::move(parts[0]), std::move(b));
    }
    return merges;
}

absl::StatusOr<MergeMap> convert_merges_to_hashmap(const Vocab& vocab, const MergeList& merges) {
    MergeMap map;
    map.reserve(merges.size());
    std::size_t skipped = 0;
    for (std::size_t rank = 0; rank < merges.size(); ++rank) {
        const auto& [a_str, b_str] = merges[rank];
        auto ia = vocab.find(a_str);
        auto ib = vocab.find(b_str);
        if (ia == vocab.end() || ib == vocab.end()) {
            ++skipped;
            continue;
        }
        Pair p{ia->second, ib->second};
        // merged token = a + b(注意:prefix/suffix 的处理在 build 时做)
        std::string merged_str = a_str + b_str;
        auto im = vocab.find(merged_str);
        if (im == vocab.end()) {
            // merged token 不在 vocab 中,无法记录;跳过(理论上不应发生在合规词表)
            ++skipped;
            continue;
        }
        map[p] = MergeValue{
            .rank = static_cast<uint32_t>(rank),
            .merged_id = im->second,
        };
    }
    if (merges.size() > 0 && skipped == merges.size()) {
        return absl::InvalidArgumentError("all merges were skipped; vocab/merges mismatch?");
    }
    return map;
}

}  // namespace bpe::bpe_internal
