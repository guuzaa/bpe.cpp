// pretokenizers/regex_engine.cpp — PCRE2 封装实现
//
// PCRE2 字符串宽度的约定:本封装只处理 UTF-8,所有 offset 均为字节偏移
// (PCRE2_UTF 模式下 ovector 给的也是字节偏移,不是字符偏移)。
//
// JIT 线程安全:同一 pcre2_code 可被多线程并发 pcre2_jit_match,
// 但 match_data 必须每线程独立。这里用 thread_local 持有 match_data,
// 并按 code 指针缓存(支持多个 Regex 实例在同一线程共存)。
#include "pretokenizers/regex_engine.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <cstring>
#include <pcre2.h>

namespace bpe::pretokenizers {

namespace {

struct CompiledCode {
    pcre2_code* code = nullptr;
    pcre2_match_data* match_data = nullptr;
    bool jit = false;  // JIT 是否可用
};

// 释放函数,用于 unique_ptr 自定义删除器
void free_code(void* p) {
    auto* c = static_cast<CompiledCode*>(p);
    if (c->match_data) {
        pcre2_match_data_free_8(c->match_data);
    }
    if (c->code) {
        pcre2_code_free_8(c->code);
    }
    delete c;
}

}  // namespace

Regex::Regex(std::string_view pattern) {
    int errcode = 0;
    PCRE2_SIZE erroff = 0;
    auto* code = pcre2_compile_8(reinterpret_cast<PCRE2_SPTR8>(pattern.data()), static_cast<PCRE2_SIZE>(pattern.size()),
                                 PCRE2_UTF | PCRE2_UCP | PCRE2_NO_UTF_CHECK, &errcode, &erroff, nullptr);
    if (!code) {
        return;
    }

    // 启用 JIT(失败不致命,回退到解释器匹配)
    int jit_rc = pcre2_jit_compile_8(code, PCRE2_JIT_COMPLETE);

    auto* c = new CompiledCode;
    c->code = code;
    c->match_data = pcre2_match_data_create_from_pattern_8(code, nullptr);
    c->jit = (jit_rc >= 0);
    code_ = static_cast<void*>(c);
}

Regex::~Regex() {
    if (code_) {
        free_code(code_);
    }
}

Regex::Regex(Regex&& o) noexcept : code_(o.code_) {
    o.code_ = nullptr;
}

Regex& Regex::operator=(Regex&& o) noexcept {
    if (this != &o) {
        if (code_) {
            free_code(code_);
        }
        code_ = o.code_;
        o.code_ = nullptr;
    }
    return *this;
}

std::vector<std::pair<std::size_t, std::size_t>> Regex::find_all(std::string_view s) const {
    std::vector<std::pair<std::size_t, std::size_t>> out;
    if (!code_ || s.empty()) {
        return out;
    }

    auto* c = static_cast<CompiledCode*>(code_);
    PCRE2_SIZE start = 0;
    const PCRE2_SIZE end = static_cast<PCRE2_SIZE>(s.size());

    while (start <= end) {
        int rc;
        if (c->jit) {
            rc = pcre2_jit_match_8(c->code, reinterpret_cast<PCRE2_SPTR8>(s.data()), end, start, PCRE2_NO_UTF_CHECK,
                                   c->match_data, nullptr);
        } else {
            rc = pcre2_match_8(c->code, reinterpret_cast<PCRE2_SPTR8>(s.data()), end, start, PCRE2_NO_UTF_CHECK,
                               c->match_data, nullptr);
        }
        if (rc < 0) {
            if (rc == PCRE2_ERROR_NOMATCH) {
                break;
            }
            // 其它错误终止
            break;
        }
        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer_8(c->match_data);
        std::size_t mstart = static_cast<std::size_t>(ovector[0]);
        std::size_t mend = static_cast<std::size_t>(ovector[1]);
        if (mend <= mstart) {
            // 空匹配:强制推进一字节,避免死循环
            ++start;
            continue;
        }
        out.emplace_back(mstart, mend);
        start = mend;
    }
    return out;
}

}  // namespace bpe::pretokenizers
