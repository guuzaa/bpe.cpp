// models/bpe/cache.h — BPE 编码侧 thread_local 无锁缓存
//
// 设计参考 §4.6 / Rust 版 model.rs:82-90。
//  - 真正的缓存是 thread_local 的两层 map:外层键为 BPE 实例 id,内层键为输入字符串
//  - 每个 BPE 实例在构造时拿一个唯一 id;clear_cache() 把旧 id 对应的内层 map
//    在本线程清掉,然后 bump id,使其它线程的旧条目自然失效(下次 miss 时重建)
//  - 容量上限按设计:DEFAULT_CAPACITY = 10000,MAX_LENGTH = 256(超长不入缓存)
#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "models/bpe/word.h"

namespace bpe::bpe_internal {

class BpeCache {
   public:
    static constexpr std::size_t kDefaultCapacity = 10000;
    static constexpr std::size_t kMaxLength = 256;

    explicit BpeCache(std::size_t capacity = kDefaultCapacity) : capacity_(capacity) {
        static std::atomic<uint64_t> next_id{1};
        id_ = next_id.fetch_add(1, std::memory_order_relaxed);
    }

    // 命中返回缓存的 Word 副本(Word 是值类型,拷贝便宜);未命中返回 nullptr
    // 注意:为避免缓存被外部修改污染,命中时返回 *拷贝* 而非引用/shared_ptr
    std::unique_ptr<Word> get(std::string_view key) const;

    // 插入:超长(> kMaxLength)或缓存已满则跳过
    void put(std::string key, std::unique_ptr<Word> w) const;

    void clear();

    std::size_t capacity() const noexcept {
        return capacity_;
    }

    void set_capacity(std::size_t n) noexcept {
        capacity_ = n;
    }

   private:
    struct Inner {
        absl::flat_hash_map<std::string, std::unique_ptr<Word>> map;
    };

    static absl::flat_hash_map<uint64_t, Inner>& local_storage();

    uint64_t id_;
    std::size_t capacity_;
};

}  // namespace bpe::bpe_internal
