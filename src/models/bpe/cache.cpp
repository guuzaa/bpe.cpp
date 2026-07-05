// models/bpe/cache.cpp
#include "models/bpe/cache.h"

namespace bpe::bpe_internal {

absl::flat_hash_map<uint64_t, BpeCache::Inner>& BpeCache::local_storage() {
    thread_local absl::flat_hash_map<uint64_t, Inner> storage;
    return storage;
}

std::unique_ptr<Word> BpeCache::get(std::string_view key) const {
    auto& storage = local_storage();
    auto it = storage.find(id_);
    if (it == storage.end()) {
        return nullptr;
    }
    auto jit = it->second.map.find(key);
    if (jit == it->second.map.end()) {
        return nullptr;
    }
    // 拷贝一份,避免调用方修改污染缓存
    return std::make_unique<Word>(*jit->second);
}

void BpeCache::put(std::string key, std::unique_ptr<Word> w) const {
    if (key.size() > kMaxLength) {
        return;
    }
    auto& storage = local_storage();
    auto& inner = storage[id_];
    if (inner.map.size() >= capacity_) {
        return;  // 满,跳过(无淘汰,同 Rust)
    }
    inner.map.emplace(std::move(key), std::move(w));
}

void BpeCache::clear() {
    auto& storage = local_storage();
    storage.erase(id_);
    // bump id 以让其它线程旧条目失效:重新分配一个新 id
    static std::atomic<uint64_t> next_id{1};
    id_ = next_id.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace bpe::bpe_internal
