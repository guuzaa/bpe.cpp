// pretokenizers/regex_engine.h — PCRE2 薄封装
//
// 设计参考 §7.1 / 第10节风险点:
//  - 模式在 Regex 构造时一次性编译,启用 PCRE2_UTF | PCRE2_UCP | JIT
//  - match_data 为 thread_local(每线程独立),避免 JIT match 时的并发冲突
//  - 提供 find_all_iter:贪心扫描整个字符串,返回所有非重叠匹配的 (start, end) 字节偏移
//  - 仅封装 BPE 所需的最小功能,不暴露 PCRE2 类型
#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

namespace bpe::pretokenizers {

class Regex {
   public:
    // 编译模式;UTF | UCP 自动启用;JIT 启用(若 PCRE2 编译时支持)
    explicit Regex(std::string_view pattern);
    ~Regex();
    Regex(const Regex&) = delete;
    Regex& operator=(const Regex&) = delete;
    Regex(Regex&&) noexcept;
    Regex& operator=(Regex&&) noexcept;

    bool valid() const noexcept {
        return code_ != nullptr;
    }

    // 贪心扫描:返回所有非重叠匹配的 [start, end) 字节偏移
    // 空匹配不会产出(避免无限循环);若模式能匹配空串,仍按 PCRE2 行为推进
    std::vector<std::pair<std::size_t, std::size_t>> find_all(std::string_view s) const;

   private:
    void* code_ = nullptr;  // pcre2_code_8*
};

}  // namespace bpe::pretokenizers
