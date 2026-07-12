// core/tokenizer.cpp — Tokenizer 总线实现
#include <algorithm>
#include <fstream>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "bpe/bpe.h"
#include "bpe/decoders.h"
#include "bpe/normalizers.h"
#include "bpe/pretokenizers.h"
#include "core/tokenizer_impl.h"
#include "nlohmann/json.hpp"
#include "normalizers/internal.h"
#include "pretokenizers/byte_level.h"
#include "pretokenizers/sequence.h"
#include "pretokenizers/split.h"

namespace bpe {

namespace {

// 默认 normalizer(Identity)的本地实现,避免引入 normalizers 头依赖
class IdentityNormalizerLocal : public Normalizer {
   public:
    void normalize(NormalizedString& s) const override {
        (void)s;
    }
};

// 解析 Split JSON 子项:`{type:"Split", pattern:{Regex|String:...}, behavior, invert}`
// 返回 Split PreTokenizer;若 pattern 缺失或为空则返回 nullptr
std::unique_ptr<PreTokenizer> parse_split_pretokenizer(const nlohmann::json& p) {
    if (!p.contains("pattern") || !p["pattern"].is_object()) {
        return nullptr;
    }
    const auto& pat = p["pattern"];
    std::string pattern_str;
    SplitPatternKind kind = SplitPatternKind::kRegex;
    if (pat.contains("Regex") && pat["Regex"].is_string()) {
        pattern_str = pat["Regex"].get<std::string>();
        kind = SplitPatternKind::kRegex;
    } else if (pat.contains("String") && pat["String"].is_string()) {
        pattern_str = pat["String"].get<std::string>();
        kind = SplitPatternKind::kString;
    } else {
        return nullptr;
    }
    if (pattern_str.empty()) {
        return nullptr;
    }
    std::string behavior_str =
        (p.contains("behavior") && p["behavior"].is_string()) ? p["behavior"].get<std::string>() : "Isolated";
    bool invert = (p.contains("invert") && p["invert"].is_boolean()) ? p["invert"].get<bool>() : false;
    return make_split_pretokenizer(std::move(pattern_str), kind, parse_split_behavior(behavior_str), invert);
}

// SplitBehavior → 字符串(与 HF JSON 一致)
const char* split_behavior_string(pretokenizers::SplitBehavior b) {
    switch (b) {
        case pretokenizers::SplitBehavior::kRemoved:
            return "Removed";
        case pretokenizers::SplitBehavior::kIsolated:
            return "Isolated";
        case pretokenizers::SplitBehavior::kMergedWithPrevious:
            return "MergedWithPrevious";
        case pretokenizers::SplitBehavior::kMergedWithNext:
            return "MergedWithNext";
        case pretokenizers::SplitBehavior::kContiguous:
            return "Contiguous";
    }
    return "Isolated";
}

// 序列化单个 PreTokenizer 为 JSON;未知类型返回 nlohmann::json(nullptr)
nlohmann::json serialize_pretokenizer(const PreTokenizer& pt) {
    if (auto* bl = dynamic_cast<const pretokenizers::ByteLevel*>(&pt)) {
        return {{"type", "ByteLevel"},
                {"add_prefix_space", bl->add_prefix_space()},
                {"trim_offsets", bl->trim_offsets()},
                {"use_regex", bl->use_regex()}};
    }
    if (auto* sp = dynamic_cast<const pretokenizers::Split*>(&pt)) {
        nlohmann::json pat = (sp->pattern_kind() == pretokenizers::SplitPatternKind::kRegex)
                                 ? nlohmann::json{{"Regex", sp->pattern()}}
                                 : nlohmann::json{{"String", sp->pattern()}};
        return {{"type", "Split"},
                {"pattern", std::move(pat)},
                {"behavior", split_behavior_string(sp->behavior())},
                {"invert", sp->invert()}};
    }
    return nullptr;
}

}  // namespace

Tokenizer::Tokenizer() : impl_(std::make_unique<Impl>()) {
}

Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

Tokenizer& Tokenizer::set_normalizer(std::unique_ptr<Normalizer> n) {
    impl_->normalizer = std::move(n);
    return *this;
}

Tokenizer& Tokenizer::set_pretokenizer(std::unique_ptr<PreTokenizer> p) {
    impl_->pretokenizer = std::move(p);
    return *this;
}

Tokenizer& Tokenizer::set_model(std::unique_ptr<Model> m) {
    impl_->model = std::move(m);
    return *this;
}

Tokenizer& Tokenizer::set_post_processor(std::unique_ptr<PostProcessor> p) {
    impl_->post_processor = std::move(p);
    return *this;
}

Tokenizer& Tokenizer::set_decoder(std::unique_ptr<Decoder> d) {
    impl_->decoder = std::move(d);
    return *this;
}

Tokenizer& Tokenizer::set_truncation(TruncationOptions opts) {
    impl_->truncation = opts;
    return *this;
}

Tokenizer& Tokenizer::set_padding(PaddingOptions opts) {
    impl_->padding = opts;
    return *this;
}

Model* Tokenizer::model() const noexcept {
    return impl_->model.get();
}

const Normalizer* Tokenizer::normalizer() const noexcept {
    return impl_->normalizer.get();
}

Decoder* Tokenizer::decoder() const noexcept {
    return impl_->decoder.get();
}

// ---- 单条 encode -----------------------------------------------------------
Encoding Tokenizer::encode(std::string_view text, bool add_special_tokens) const {
    return impl_->encode_one(text, add_special_tokens);
}

Encoding Tokenizer::Impl::encode_one(std::string_view text, bool add_special_tokens) const {
    Encoding out;
    if (!model) {
        return out;
    }

    // 1. Normalize
    NormalizedString ns{std::string(text)};
    if (normalizer) {
        normalizer->normalize(ns);
    }

    // 2. Pre-tokenize:得到若干 PreToken(text + offsets 是 normalized 偏移)
    std::vector<PreToken> pre_tokens;
    if (pretokenizer) {
        PreTokenizedString ps{std::vector<PreToken>{PreToken{
            std::string(ns.get()), std::pair<uint32_t, uint32_t>{0, static_cast<uint32_t>(ns.get().size())}, {}}}};
        pretokenizer->pre_tokenize(ps);
        pre_tokens = std::move(ps.tokens());
    } else {
        pre_tokens.push_back(PreToken{
            std::string(ns.get()), std::pair<uint32_t, uint32_t>{0, static_cast<uint32_t>(ns.get().size())}, {}});
    }

    // 3. Model::tokenize 每个 pre-token,收集 Token
    //    Model 返回的 offsets 是 pt.text 内的相对字节偏移
    //    若 pt.alignment 非空(ByteLevel),需用它转换回 normalized 空间
    std::vector<Token> tokens;
    for (auto& pt : pre_tokens) {
        auto sub = model->tokenize(pt.text);
        for (auto& t : sub) {
            Token nt = t;
            if (!pt.alignment.empty()) {
                // ByteLevel:byte-char 字节偏移 → 原始段内字节偏移 → normalized 偏移
                auto clamp = [&](uint32_t i) -> uint32_t {
                    if (i >= pt.alignment.size()) {
                        return pt.alignment.empty() ? i : pt.alignment.back();
                    }
                    return pt.alignment[i];
                };
                uint32_t rel_start = clamp(t.offsets.first);
                uint32_t rel_end = (t.offsets.second > 0) ? clamp(t.offsets.second - 1) + 1 : rel_start;
                if (rel_end < rel_start) {
                    rel_end = rel_start;
                }
                nt.offsets = {pt.offsets.first + rel_start, pt.offsets.first + rel_end};
            } else {
                // 非 ByteLevel:text 字节偏移直接 == 原始段字节偏移
                nt.offsets = {
                    pt.offsets.first + t.offsets.first,
                    pt.offsets.first + t.offsets.second,
                };
            }
            tokens.push_back(std::move(nt));
        }
    }

    // 4. Post-process
    if (post_processor) {
        tokens = post_processor->process(std::move(tokens));
    }
    (void)add_special_tokens;  // M6 不实现 special token 模板

    // 5. 转换到 normalized 偏移对齐到 original(由 NormalizedString)
    for (auto& t : tokens) {
        auto orig = ns.align_to_original(t.offsets.first, t.offsets.second);
        t.offsets = orig;
    }

    // 6. 装填 Encoding
    out.ids.reserve(tokens.size());
    out.tokens.reserve(tokens.size());
    out.offsets.reserve(tokens.size());
    out.attention_mask.assign(tokens.size(), 1);
    out.special_tokens_mask.assign(tokens.size(), 0);
    out.type_ids.assign(tokens.size(), 0);
    for (auto& t : tokens) {
        out.ids.push_back(t.id);
        out.tokens.push_back(std::move(t.value));
        out.offsets.push_back(t.offsets);
    }

    // 7. Truncation
    if (truncation) {
        apply_truncation(out);
    }

    return out;
}

void Tokenizer::Impl::apply_truncation(Encoding& e) const {
    if (!truncation || !truncation->max_length) {
        return;
    }
    std::size_t max_len = *truncation->max_length;
    if (e.ids.size() <= max_len) {
        return;
    }
    if (truncation->direction_from_end) {
        // 从尾部保留
        e.ids.erase(e.ids.begin(), e.ids.begin() + static_cast<std::ptrdiff_t>(e.ids.size() - max_len));
        e.tokens.erase(e.tokens.begin(), e.tokens.begin() + static_cast<std::ptrdiff_t>(e.tokens.size() - max_len));
        e.offsets.erase(e.offsets.begin(), e.offsets.begin() + static_cast<std::ptrdiff_t>(e.offsets.size() - max_len));
        e.attention_mask.assign(max_len, 1);
        e.special_tokens_mask.assign(max_len, 0);
        e.type_ids.assign(max_len, 0);
    } else {
        e.ids.resize(max_len);
        e.tokens.resize(max_len);
        e.offsets.resize(max_len);
        e.attention_mask.assign(max_len, 1);
        e.special_tokens_mask.assign(max_len, 0);
        e.type_ids.assign(max_len, 0);
    }
}

void Tokenizer::Impl::apply_padding(std::vector<Encoding>& batch) const {
    if (!padding) {
        return;
    }
    std::size_t target = 0;
    if (padding->max_length) {
        target = *padding->max_length;
    } else {
        for (const auto& e : batch) {
            target = std::max(target, e.ids.size());
        }
    }
    for (auto& e : batch) {
        if (e.ids.size() >= target) {
            continue;
        }
        std::size_t pad = target - e.ids.size();
        e.ids.insert(e.ids.end(), pad, padding->pad_id);
        e.tokens.insert(e.tokens.end(), pad, padding->pad_token);
        e.offsets.insert(e.offsets.end(), pad, {0, 0});
        e.attention_mask.insert(e.attention_mask.end(), pad, 0);
        e.special_tokens_mask.insert(e.special_tokens_mask.end(), pad, 0);
        e.type_ids.insert(e.type_ids.end(), pad, padding->pad_type_id);
    }
}

// ---- batch encode ----------------------------------------------------------
std::vector<Encoding> Tokenizer::encode_batch(const std::vector<std::string>& texts, bool add_special_tokens) const {
    std::vector<Encoding> out;
    out.reserve(texts.size());
    for (const auto& t : texts) {
        out.push_back(impl_->encode_one(t, add_special_tokens));
    }
    impl_->apply_padding(out);
    return out;
}

// ---- decode ----------------------------------------------------------------
std::string Tokenizer::decode(const std::vector<TokenId>& ids, bool skip_special_tokens) const {
    if (!impl_->decoder || !impl_->model) {
        return {};
    }
    // 把 id → token string(走 model.id_to_token)
    std::vector<std::string> token_strs;
    token_strs.reserve(ids.size());
    for (auto id : ids) {
        auto t = impl_->model->id_to_token(id);
        if (!t) {
            continue;
        }
        // M6:简单按 special token 名字过滤
        // 真实实现需要 AddedVocabulary 维护 special token 集合;此处按 <...> 形态粗过滤
        if (skip_special_tokens && !t->empty() && t->front() == '<' && t->back() == '>') {
            continue;
        }
        token_strs.push_back(*std::move(t));
    }
    return impl_->decoder->decode(token_strs);
}

// ---- 文件 I/O --------------------------------------------------------------
absl::StatusOr<std::unique_ptr<Tokenizer>> Tokenizer::from_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        return absl::NotFoundError(absl::StrCat("cannot open ", path));
    }
    nlohmann::json j;
    try {
        ifs >> j;
    } catch (const std::exception& e) {
        return absl::InvalidArgumentError(absl::StrCat("parse tokenizer.json failed: ", e.what()));
    }

    auto tok = std::make_unique<Tokenizer>();

    // model 段
    if (j.contains("model")) {
        auto& m = j["model"];
        std::string type = (m.contains("type") && m["type"].is_string()) ? m["type"].get<std::string>() : "BPE";
        if (type != "BPE") {
            return absl::UnimplementedError(absl::StrCat("unsupported model type: ", type));
        }
        Vocab vocab;
        if (m.contains("vocab")) {
            for (auto it = m["vocab"].begin(); it != m["vocab"].end(); ++it) {
                vocab[it.key()] = static_cast<TokenId>(it.value().get<int64_t>());
            }
        }
        MergeList merges;
        if (m.contains("merges")) {
            for (const auto& mr : m["merges"]) {
                if (mr.is_array() && mr.size() == 2) {
                    merges.emplace_back(mr[0].get<std::string>(), mr[1].get<std::string>());
                } else if (mr.is_string()) {
                    // 兼容旧格式 "a b"
                    std::string s = mr.get<std::string>();
                    std::vector<std::string> parts = absl::StrSplit(s, ' ', absl::SkipEmpty());
                    if (parts.size() == 2) {
                        merges.emplace_back(parts[0], parts[1]);
                    }
                }
            }
        }
        BpeOptions opts;
        // 注意:字段值可能为 JSON null(如 deepseek 的 unk_token = null),
        // 需先 is_string/is_number 再 get
        if (m.contains("unk_token") && m["unk_token"].is_string()) {
            opts.unk_token = m["unk_token"].get<std::string>();
        }
        if (m.contains("continuing_subword_prefix") && m["continuing_subword_prefix"].is_string()) {
            opts.continuing_subword_prefix = m["continuing_subword_prefix"].get<std::string>();
        }
        if (m.contains("end_of_word_suffix") && m["end_of_word_suffix"].is_string()) {
            opts.end_of_word_suffix = m["end_of_word_suffix"].get<std::string>();
        }
        if (m.contains("fuse_unk") && m["fuse_unk"].is_boolean()) {
            opts.fuse_unk = m["fuse_unk"].get<bool>();
        }
        if (m.contains("byte_fallback") && m["byte_fallback"].is_boolean()) {
            opts.byte_fallback = m["byte_fallback"].get<bool>();
        }
        if (m.contains("ignore_merges") && m["ignore_merges"].is_boolean()) {
            opts.ignore_merges = m["ignore_merges"].get<bool>();
        }
        if (m.contains("dropout") && m["dropout"].is_number()) {
            opts.dropout = m["dropout"].get<float>();
        }

        // 用 BpeBuilder 一次性传入 vocab/merges 与所有 opts
        auto b = std::make_unique<BpeBuilder>();
        b->vocab_and_merges(std::move(vocab), std::move(merges)).cache_capacity(10000);
        if (opts.unk_token) {
            b->unk_token(*opts.unk_token);
        }
        if (opts.continuing_subword_prefix) {
            b->continuing_subword_prefix(*opts.continuing_subword_prefix);
        }
        if (opts.end_of_word_suffix) {
            b->end_of_word_suffix(*opts.end_of_word_suffix);
        }
        if (opts.fuse_unk) {
            b->fuse_unk(true);
        }
        if (opts.byte_fallback) {
            b->byte_fallback(true);
        }
        if (opts.ignore_merges) {
            b->ignore_merges(true);
        }
        if (opts.dropout) {
            b->dropout(*opts.dropout);
        }

        auto bpe_status = b->build();
        if (!bpe_status.ok()) {
            return bpe_status.status();
        }
        tok->set_model(std::move(*bpe_status));
    }

    // normalizer 段(可选):仅支持 sequence / lowercase / uppercase / strip / identity
    // 注意:字段值可能为 JSON null(如 glm 的 normalizer = null),需跳过
    if (j.contains("normalizer") && !j["normalizer"].is_null()) {
        auto& n = j["normalizer"];
        std::string type = (n.contains("type") && n["type"].is_string()) ? n["type"].get<std::string>() : "Sequence";
        if (type == "Sequence" && n.contains("normalizers") && n["normalizers"].is_array()) {
            std::vector<std::unique_ptr<Normalizer>> ns;
            ns.reserve(n["normalizers"].size());
            for (auto& sub : n["normalizers"]) {
                std::string st =
                    (sub.contains("type") && sub["type"].is_string()) ? sub["type"].get<std::string>() : "Identity";
                ns.push_back(make_normalizer(parse_normalizer_kind(st)));
            }
            tok->set_normalizer(make_sequence_normalizer(std::move(ns)));
        } else {
            tok->set_normalizer(make_normalizer(parse_normalizer_kind(type)));
        }
    }

    // pre_tokenizer 段:
    //   - ByteLevel → 直接使用
    //   - Split     → 直接构造单个 Split
    //   - Sequence  → 顺序组合子项(Split / ByteLevel / 其它支持的);此时
    //                 内部 ByteLevel 的 use_regex 按其原始配置(DeepSeek 等为 false)
    //   - null / 其它 → 不设置
    if (j.contains("pre_tokenizer") && !j["pre_tokenizer"].is_null()) {
        auto& p = j["pre_tokenizer"];
        std::string type = (p.contains("type") && p["type"].is_string()) ? p["type"].get<std::string>() : "ByteLevel";

        if (type == "Split") {
            tok->set_pretokenizer(parse_split_pretokenizer(p));
        } else if (type == "Sequence" && p.contains("pretokenizers") && p["pretokenizers"].is_array()) {
            std::vector<std::unique_ptr<PreTokenizer>> subs;
            subs.reserve(p["pretokenizers"].size());
            for (auto& sub : p["pretokenizers"]) {
                std::string st =
                    (sub.contains("type") && sub["type"].is_string()) ? sub["type"].get<std::string>() : "";
                if (st == "Split") {
                    auto sp = parse_split_pretokenizer(sub);
                    if (sp) {
                        subs.push_back(std::move(sp));
                    }
                } else if (st == "ByteLevel") {
                    bool add_prefix = (sub.contains("add_prefix_space") && sub["add_prefix_space"].is_boolean())
                                          ? sub["add_prefix_space"].get<bool>()
                                          : true;
                    bool trim_off = (sub.contains("trim_offsets") && sub["trim_offsets"].is_boolean())
                                        ? sub["trim_offsets"].get<bool>()
                                        : true;
                    bool use_re = (sub.contains("use_regex") && sub["use_regex"].is_boolean())
                                      ? sub["use_regex"].get<bool>()
                                      : true;
                    subs.push_back(make_byte_level_pretokenizer(add_prefix, trim_off, use_re));
                }
                // 其它未知子项被忽略;如全部被忽略则 fallback 到下一阶段默认逻辑
            }
            if (!subs.empty()) {
                tok->set_pretokenizer(make_sequence_pretokenizer(std::move(subs)));
            } else {
                tok->set_pretokenizer(make_byte_level_pretokenizer(true, true, true));
            }
        } else if (type == "ByteLevel") {
            bool add_prefix = (p.contains("add_prefix_space") && p["add_prefix_space"].is_boolean())
                                  ? p["add_prefix_space"].get<bool>()
                                  : true;
            bool trim_off =
                (p.contains("trim_offsets") && p["trim_offsets"].is_boolean()) ? p["trim_offsets"].get<bool>() : true;
            bool use_re = (p.contains("use_regex") && p["use_regex"].is_boolean()) ? p["use_regex"].get<bool>() : true;
            tok->set_pretokenizer(make_byte_level_pretokenizer(add_prefix, trim_off, use_re));
        }
    }

    // decoder 段:仅支持 ByteLevel / BPEDecoder(默认 ByteLevel)
    if (j.contains("decoder") && !j["decoder"].is_null()) {
        auto& d = j["decoder"];
        std::string type = (d.contains("type") && d["type"].is_string()) ? d["type"].get<std::string>() : "ByteLevel";
        if (type == "ByteLevel") {
            tok->set_decoder(make_byte_level_decoder());
        }
    } else {
        // 默认 ByteLevel decoder
        tok->set_decoder(make_byte_level_decoder());
    }

    // post_processor 段:ByteLevel(直接 new,因为 ByteLevel 同时实现 PostProcessor)
    if (j.contains("post_processor") && !j["post_processor"].is_null()) {
        auto& p = j["post_processor"];
        std::string type = (p.contains("type") && p["type"].is_string()) ? p["type"].get<std::string>() : "ByteLevel";
        if (type == "ByteLevel") {
            bool add_prefix = (p.contains("add_prefix_space") && p["add_prefix_space"].is_boolean())
                                  ? p["add_prefix_space"].get<bool>()
                                  : true;
            bool trim_off =
                (p.contains("trim_offsets") && p["trim_offsets"].is_boolean()) ? p["trim_offsets"].get<bool>() : true;
            tok->set_post_processor(std::make_unique<pretokenizers::ByteLevel>(add_prefix, trim_off, true));
        }
    }

    return tok;
}

absl::Status Tokenizer::to_file(const std::string& path) const {
    nlohmann::json j;
    j["version"] = "1.0";

    // model 段:仅支持 BPE
    if (impl_->model) {
        nlohmann::json m;
        m["type"] = "BPE";
        // 取 vocab(BPE 公开了 vocab())
        auto* bpe = dynamic_cast<BPE*>(impl_->model.get());
        if (bpe) {
            nlohmann::json v = nlohmann::json::object();
            for (const auto& [tok, id] : bpe->vocab()) {
                v[tok] = id;
            }
            m["vocab"] = v;
            // merges:从 BPE::merges_list() 取,写为 [["a","b"],...] 格式
            auto ml = bpe->merges_list();
            nlohmann::json merges_arr = nlohmann::json::array();
            for (const auto& [a, b] : ml) {
                merges_arr.push_back({a, b});
            }
            m["merges"] = merges_arr;
            const auto& opts = bpe->options();
            if (opts.unk_token) {
                m["unk_token"] = *opts.unk_token;
            }
            if (opts.continuing_subword_prefix) {
                m["continuing_subword_prefix"] = *opts.continuing_subword_prefix;
            }
            if (opts.end_of_word_suffix) {
                m["end_of_word_suffix"] = *opts.end_of_word_suffix;
            }
            m["fuse_unk"] = opts.fuse_unk;
            m["byte_fallback"] = opts.byte_fallback;
            m["ignore_merges"] = opts.ignore_merges;
        }
        j["model"] = m;
    }
    if (impl_->normalizer) {
        j["normalizer"] = nlohmann::json::object({{"type", "Identity"}});
    }

    // pre_tokenizer:支持 ByteLevel / Split / Sequence(Split + ByteLevel ...)
    if (impl_->pretokenizer) {
        auto* bl = dynamic_cast<pretokenizers::ByteLevel*>(impl_->pretokenizer.get());
        if (bl) {
            j["pre_tokenizer"] = {{"type", "ByteLevel"},
                                  {"add_prefix_space", bl->add_prefix_space()},
                                  {"trim_offsets", bl->trim_offsets()},
                                  {"use_regex", bl->use_regex()}};
        } else if (auto* sp = dynamic_cast<pretokenizers::Split*>(impl_->pretokenizer.get())) {
            j["pre_tokenizer"] = serialize_pretokenizer(*sp);
        } else if (auto* seq = dynamic_cast<pretokenizers::Sequence*>(impl_->pretokenizer.get())) {
            nlohmann::json arr = nlohmann::json::array();
            for (const auto& sub : seq->pretokenizers()) {
                if (!sub) {
                    continue;
                }
                auto sub_json = serialize_pretokenizer(*sub);
                if (!sub_json.is_null()) {
                    arr.push_back(std::move(sub_json));
                }
            }
            j["pre_tokenizer"] = {{"type", "Sequence"}, {"pretokenizers", std::move(arr)}};
        }
    }
    if (impl_->decoder) {
        j["decoder"] = nlohmann::json::object({{"type", "ByteLevel"}});
    }

    // post_processor:从实际 ByteLevel 实例读取配置
    if (impl_->post_processor) {
        auto* bl = dynamic_cast<pretokenizers::ByteLevel*>(impl_->post_processor.get());
        if (bl) {
            j["post_processor"] = {{"type", "ByteLevel"},
                                   {"add_prefix_space", bl->add_prefix_space()},
                                   {"trim_offsets", bl->trim_offsets()},
                                   {"use_regex", bl->use_regex()}};
        }
    }

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        return absl::PermissionDeniedError(absl::StrCat("cannot write ", path));
    }
    ofs << j.dump(2);
    return absl::OkStatus();
}

absl::Status Tokenizer::to_vocab_merges(const std::string& dir) const {
    auto* bpe = dynamic_cast<BPE*>(impl_->model.get());
    if (!bpe) {
        return absl::FailedPreconditionError("model is not BPE");
    }
    // vocab.json
    nlohmann::json v = nlohmann::json::object();
    for (const auto& [tok, id] : bpe->vocab()) {
        v[tok] = id;
    }
    std::ofstream vfs(dir + "/vocab.json");
    if (!vfs.is_open()) {
        return absl::PermissionDeniedError(absl::StrCat("cannot write ", dir, "/vocab.json"));
    }
    vfs << v.dump(2);
    // merges.txt:M6 暂不暴露,写空(后续可加 BPE::merges_list())
    std::ofstream mfs(dir + "/merges.txt");
    if (!mfs.is_open()) {
        return absl::PermissionDeniedError(absl::StrCat("cannot write ", dir, "/merges.txt"));
    }
    mfs << "#version: 0.1\n";
    return absl::OkStatus();
}

}  // namespace bpe