# qbitos-multilingual

Multilingual processing for QbitOS: language detection, tokenization, stemming, and low-resource translation.

## Modules

| Module | Description |
|--------|-------------|
| `detection` | Language detection via n-gram profile matching |
| `tokenize` | Word and sentence tokenization |
| `stem` | Snowball-style stemming |
| `normalize` | Unicode normalization, diacritic stripping |
| `transliterate` | Cyrillic ↔ Latin transliteration |
| `low_resource_translation` | Few-shot translation with back-translation verification (WMT 2026) |

## License

Apache-2.0
