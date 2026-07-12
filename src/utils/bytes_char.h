// utils/bytes_char.h — GPT-2 的 256 项 byte↔unicode char 映射表
//
// 设计参考 §7.1 / HF tokenizers byte_level.rs:15-39。
// 规则:
//  - 可打印 ASCII: 0x21..0x7E   → 原字节
//  - 拉丁 1 段:    0xA1..0xAC, 0xAE..0xFF → 原字节
//  - 其余 68 个控制字节 → U+0100..U+0143(空闲区),使得所有字节都是可打印字符
#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace bpe::util {

// byte → unicode code point
const std::array<uint32_t, 256>& bytes_to_chars() noexcept;

// code point → byte(若不在映射表中返回 -1)
// 用静态查表实现,首次调用时构建反向表
int char_to_byte(uint32_t cp) noexcept;

// 把一个 byte 编码为 UTF-8 字符串(对应 HF 的 BYTES_CHAR[b])
std::string byte_to_string(uint8_t b);

// 把一个 byte 对应的 byte-char UTF-8 追加到 out(热路径,避免临时 string)
void append_byte_char(std::string& out, uint8_t b);

}  // namespace bpe::util
