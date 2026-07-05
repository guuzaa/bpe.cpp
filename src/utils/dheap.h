// utils/dheap.h — 简易 d-ary 堆
//
// 设计目的:替换 std::priority_queue,用 d 叉堆降低比较次数。
// 编码用 D=4(对应 Rust 版 QuaternaryHeap),训练用 D=8(OctonaryHeap)。
// 仅提供 push/pop/top/empty/clear,接口与 std::priority_queue 类似。
// T 必须满足 Compare(默认 std::less<T>,即最大堆行为)。
#pragma once

#include <functional>
#include <utility>
#include <vector>

namespace bpe::util {

template <typename T, std::size_t D = 4, typename Compare = std::less<T>>
class d_heap {
    static_assert(D >= 2, "d-ary heap requires D >= 2");

   public:
    d_heap() = default;

    explicit d_heap(Compare cmp) : cmp_(std::move(cmp)) {
    }

    void push(T v) {
        data_.push_back(std::move(v));
        sift_up(data_.size() - 1);
    }

    void pop() {
        if (data_.empty()) {
            return;
        }
        if (data_.size() > 1) {
            data_.front() = std::move(data_.back());
        }
        data_.pop_back();
        if (!data_.empty()) {
            sift_down(0);
        }
    }

    const T& top() const {
        return data_.front();
    }

    bool empty() const noexcept {
        return data_.empty();
    }

    std::size_t size() const noexcept {
        return data_.size();
    }

    void clear() {
        data_.clear();
    }

   private:
    void sift_up(std::size_t i) {
        while (i > 0) {
            std::size_t p = (i - 1) / D;
            if (!cmp_(data_[p], data_[i])) {
                break;
            }

            std::swap(data_[p], data_[i]);
            i = p;
        }
    }

    void sift_down(std::size_t i) {
        const std::size_t n = data_.size();
        for (;;) {
            std::size_t best = i;
            std::size_t base = i * D + 1;
            for (std::size_t k = 0; k < D; ++k) {
                std::size_t c = base + k;
                if (c >= n) {
                    break;
                }
                if (cmp_(data_[best], data_[c])) {
                    best = c;
                }
            }
            if (best == i) {
                break;
            }
            std::swap(data_[i], data_[best]);
            i = best;
        }
    }

    std::vector<T> data_{};
    Compare cmp_{};
};

}  // namespace bpe::util
