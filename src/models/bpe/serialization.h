// models/bpe/serialization.h — vocab.json / merges.txt 加载
//
// 设计参考 §8.1:
//  - vocab.json:JSON object {token: id}
//  - merges.txt:每行 "a b",行号即 rank
//  - convert_merges_to_hashmap:把 (a,b) 字符串对解析为 MergeMap,需查 vocab 拿 id
#pragma once

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "bpe/types.h"
#include "models/bpe/word.h"  // MergeValue

namespace bpe::bpe_internal {

// 内部使用的 MergeMap 类型:absl::flat_hash_map,自动用 AbslHashValue<Pair> 哈希
using MergeMap = absl::flat_hash_map<Pair, MergeValue>;

// 读取 vocab.json → Vocab(token -> id)
absl::StatusOr<Vocab> read_vocab_json(const std::string& path);

// 读取 merges.txt → MergeList([(a, b), ...]);行号即 rank
absl::StatusOr<MergeList> read_merges_txt(const std::string& path);

// 把 MergeList + Vocab 转为 MergeMap;跳过 a/b 不在 vocab 中的行(同 Rust 行为)
// 返回 StatusOr 以便在全部行都无效时报错
absl::StatusOr<MergeMap> convert_merges_to_hashmap(const Vocab& vocab, const MergeList& merges);

}  // namespace bpe::bpe_internal
