// utils/parallel.h — 并行分片用的 worker 数选择
#pragma once

#include <cstddef>
#include <thread>

namespace bpe::util {

// 返回适合对 work_items 做分片的线程数:
//   - 以 hardware_concurrency 为基准(0 时回退 default_hw)
//   - 上限 cap(默认 16)
//   - 不超过 work_items,至少 1
inline unsigned decide_num_workers(std::size_t work_items, unsigned cap = 16,
                                   unsigned default_hw = 4) noexcept {
    if (work_items == 0) {
        return 1;
    }
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        hw = default_hw;
    }
    if (cap != 0 && hw > cap) {
        hw = cap;
    }
    if (hw > work_items) {
        hw = static_cast<unsigned>(work_items);
    }
    return hw < 1 ? 1u : hw;
}

}  // namespace bpe::util
