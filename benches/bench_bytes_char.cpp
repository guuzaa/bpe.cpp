// benches/bench_bytes_char.cpp
// 验证 char_to_byte 反向查表:std::unordered_map vs absl::flat_hash_map
//  - BM_lookup_*:256 个 cp 流式查询的吞吐(每秒查询数)
//  - BM_mem_*   :用计数分配器统计 256 项 map 的堆分配字节与分配次数
#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "utils/bytes_char.h"

namespace {

// 256 个 (cp → byte) 对,两种 map 共用同一份数据
const std::vector<std::pair<uint32_t, uint8_t>>& rev_pairs() {
    static const std::vector<std::pair<uint32_t, uint8_t>> v = [] {
        std::vector<std::pair<uint32_t, uint8_t>> out;
        out.reserve(256);
        const auto& fwd = bpe::util::bytes_to_chars();
        for (uint32_t b = 0; b <= 0xFFu; ++b) {
            out.emplace_back(fwd[b], static_cast<uint8_t>(b));
        }
        return out;
    }();
    return v;
}

// 查询键流:256 个 cp × 4 轮 = 1024 次查询/迭代,顺序访问两种 map 公平
const std::vector<uint32_t>& query_keys() {
    static const std::vector<uint32_t> v = [] {
        std::vector<uint32_t> out;
        out.reserve(256 * 4);
        for (int rep = 0; rep < 4; ++rep) {
            for (const auto& kv : rev_pairs()) out.push_back(kv.first);
        }
        return out;
    }();
    return v;
}

const std::unordered_map<uint32_t, uint8_t>& rev_unordered() {
    static const std::unordered_map<uint32_t, uint8_t> m = [] {
        std::unordered_map<uint32_t, uint8_t> m;
        m.reserve(256);
        for (const auto& kv : rev_pairs()) m.emplace(kv.first, kv.second);
        return m;
    }();
    return m;
}

const absl::flat_hash_map<uint32_t, uint8_t>& rev_flat() {
    static const absl::flat_hash_map<uint32_t, uint8_t> m = [] {
        absl::flat_hash_map<uint32_t, uint8_t> m;
        m.reserve(256);
        for (const auto& kv : rev_pairs()) m.emplace(kv.first, kv.second);
        return m;
    }();
    return m;
}

// ---- 查表吞吐 -------------------------------------------------------------
void BM_lookup_unordered(benchmark::State& state) {
    const auto& m = rev_unordered();
    const auto& keys = query_keys();
    int acc = 0;
    for (auto _ : state) {
        for (uint32_t cp : keys) {
            auto it = m.find(cp);
            int r = (it == m.end()) ? -1 : it->second;
            acc += r;
        }
    }
    benchmark::DoNotOptimize(acc);
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(keys.size()));
}
BENCHMARK(BM_lookup_unordered)->Unit(benchmark::kNanosecond);

void BM_lookup_flat(benchmark::State& state) {
    const auto& m = rev_flat();
    const auto& keys = query_keys();
    int acc = 0;
    for (auto _ : state) {
        for (uint32_t cp : keys) {
            auto it = m.find(cp);
            int r = (it == m.end()) ? -1 : it->second;
            acc += r;
        }
    }
    benchmark::DoNotOptimize(acc);
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(keys.size()));
}
BENCHMARK(BM_lookup_flat)->Unit(benchmark::kNanosecond);

// ---- 内存:计数分配器统计 256 项 map 的堆占用 -----------------------------
template <class T>
class CountingAlloc {
public:
    using value_type = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    CountingAlloc(std::size_t* bytes, std::size_t* allocs) noexcept
        : bytes_(bytes), allocs_(allocs) {}

    template <class U>
    CountingAlloc(const CountingAlloc<U>& o) noexcept
        : bytes_(o.bytes_), allocs_(o.allocs_) {}

    T* allocate(std::size_t n) {
        if (bytes_) *bytes_ += n * sizeof(T);
        if (allocs_) *allocs_ += 1;
        return std::allocator<T>{}.allocate(n);
    }
    void deallocate(T* p, std::size_t n) noexcept {
        std::allocator<T>{}.deallocate(p, n);
    }

    template <class U>
    bool operator==(const CountingAlloc<U>& o) const noexcept {
        return bytes_ == o.bytes_;
    }
    template <class U>
    bool operator!=(const CountingAlloc<U>& o) const noexcept {
        return bytes_ != o.bytes_;
    }

    std::size_t* bytes_ = nullptr;
    std::size_t* allocs_ = nullptr;
};

template <class MapT>
void BM_mem_common(benchmark::State& state, MapT&& factory) {
    std::size_t bytes = 0, allocs = 0;
    // 构建一次,捕获稳态堆占用;循环体仅做一次 find 以提供 benchmark 主体
    auto m = factory(&bytes, &allocs);
    const uint32_t probe = rev_pairs().front().first;
    for (auto _ : state) {
        auto it = m.find(probe);
        benchmark::DoNotOptimize(it);
    }
    state.counters["bytes"] = static_cast<double>(bytes);
    state.counters["allocs"] = static_cast<double>(allocs);
}

void BM_mem_unordered(benchmark::State& state) {
    BM_mem_common(state, [](std::size_t* b, std::size_t* a) {
        using A = CountingAlloc<std::pair<const uint32_t, uint8_t>>;
        std::unordered_map<uint32_t, uint8_t, std::hash<uint32_t>, std::equal_to<uint32_t>, A> m(A(b, a));
        m.reserve(256);
        for (const auto& kv : rev_pairs()) m.emplace(kv.first, kv.second);
        return m;
    });
}
BENCHMARK(BM_mem_unordered)->Unit(benchmark::kNanosecond);

void BM_mem_flat(benchmark::State& state) {
    BM_mem_common(state, [](std::size_t* b, std::size_t* a) {
        using A = CountingAlloc<std::pair<const uint32_t, uint8_t>>;
        absl::flat_hash_map<uint32_t, uint8_t, std::hash<uint32_t>, std::equal_to<uint32_t>, A> m(A(b, a));
        m.reserve(256);
        for (const auto& kv : rev_pairs()) m.emplace(kv.first, kv.second);
        return m;
    });
}
BENCHMARK(BM_mem_flat)->Unit(benchmark::kNanosecond);

}  // namespace
BENCHMARK_MAIN();
