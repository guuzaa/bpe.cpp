// normalizers/normalizers.cpp — Identity / Strip / Lowercase / Uppercase / Sequence
#include "bpe/normalizers.h"
#include <utility>
#include <vector>

namespace bpe {

namespace {

class IdentityNormalizer : public Normalizer {
   public:
    void normalize(NormalizedString& s) const override {
        (void)s;
    }
};

class StripNormalizer : public Normalizer {
   public:
    void normalize(NormalizedString& s) const override {
        s.strip();
    }
};

class LowercaseNormalizer : public Normalizer {
   public:
    void normalize(NormalizedString& s) const override {
        s.lowercase();
    }
};

class UppercaseNormalizer : public Normalizer {
   public:
    void normalize(NormalizedString& s) const override {
        s.uppercase();
    }
};

class SequenceNormalizer : public Normalizer {
   public:
    explicit SequenceNormalizer(std::vector<std::unique_ptr<Normalizer>> ns) : ns_(std::move(ns)) {
    }

    void normalize(NormalizedString& s) const override {
        for (const auto& n : ns_) {
            n->normalize(s);
        }
    }

   private:
    std::vector<std::unique_ptr<Normalizer>> ns_;
};

}  // namespace

std::unique_ptr<Normalizer> make_identity_normalizer() {
    return std::make_unique<IdentityNormalizer>();
}

std::unique_ptr<Normalizer> make_strip_normalizer() {
    return std::make_unique<StripNormalizer>();
}

std::unique_ptr<Normalizer> make_lowercase_normalizer() {
    return std::make_unique<LowercaseNormalizer>();
}

std::unique_ptr<Normalizer> make_uppercase_normalizer() {
    return std::make_unique<UppercaseNormalizer>();
}

std::unique_ptr<Normalizer> make_sequence_normalizer(std::vector<std::unique_ptr<Normalizer>> ns) {
    return std::make_unique<SequenceNormalizer>(std::move(ns));
}

}  // namespace bpe