//! Stop word lists for 12+ languages.

use crate::detection::Language;
use ahash::AHashSet;

/// Whether `word` is a stop word in the given language (case-insensitive).
#[must_use]
pub fn is_stopword(word: &str, lang: Language) -> bool {
    let set = stopword_set(lang);
    set.contains(&word.to_lowercase())
}

/// Get the stop word set for a language.
#[must_use]
pub fn stopword_set(lang: Language) -> AHashSet<String> {
    use std::sync::OnceLock;
    static SETS: OnceLock<AHashMap<Language, AHashSet<String>>> = OnceLock::new();
    use ahash::AHashMap;
    let sets = SETS.get_or_init(|| {
        let mut m: AHashMap<Language, AHashSet<String>> = AHashMap::new();
        m.insert(Language::English, set_from(ENGLISH_STOPWORDS));
        m.insert(Language::Spanish, set_from(SPANISH_STOPWORDS));
        m.insert(Language::French, set_from(FRENCH_STOPWORDS));
        m.insert(Language::German, set_from(GERMAN_STOPWORDS));
        m.insert(Language::Italian, set_from(ITALIAN_STOPWORDS));
        m.insert(Language::Portuguese, set_from(PORTUGUESE_STOPWORDS));
        m.insert(Language::Dutch, set_from(DUTCH_STOPWORDS));
        m.insert(Language::Russian, set_from(RUSSIAN_STOPWORDS));
        m
    });
    sets.get(&lang).cloned().unwrap_or_default()
}

/// Get the stop word list for a language as a sorted vector.
#[must_use]
pub fn stopword_list(lang: Language) -> Vec<String> {
    let mut v: Vec<String> = stopword_set(lang).iter().cloned().collect();
    v.sort();
    v
}

fn set_from(words: &[&str]) -> AHashSet<String> {
    words.iter().map(|s| (*s).to_string()).collect()
}

const ENGLISH_STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from", "had",
    "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "of", "on", "or",
    "she", "that", "the", "their", "them", "they", "this", "to", "was", "we", "were",
    "will", "with", "you", "your", "would", "could", "should", "do", "does", "did",
    "not", "no", "yes", "if", "then", "else", "when", "where", "who", "what", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "so", "than", "too", "very", "can", "just", "now",
];

const SPANISH_STOPWORDS: &[&str] = &[
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "es", "son",
    "por", "para", "con", "su", "sus", "y", "o", "que", "como", "más", "pero", "no", "sí",
    "se", "lo", "le", "les", "al", "lo", "me", "te", "nos", "os", "se", "ya", "todo",
    "todos", "toda", "todas", "cada", "todo", "este", "esta", "estos", "estas", "ese",
    "esa", "esos", "esas", "aquel", "aquella", "aquellos", "aquellas",
];

const FRENCH_STOPWORDS: &[&str] = &[
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "que", "qui", "dans",
    "pour", "pas", "ne", "ce", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
    "son", "sa", "ses", "notre", "votre", "leur", "leurs", "je", "tu", "il", "elle",
    "nous", "vous", "ils", "elles", "est", "sont", "été", "être", "avoir", "avec", "sur",
    "en", "au", "aux", "se", "se", "y", "à",
];

const GERMAN_STOPWORDS: &[&str] = &[
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines", "einem",
    "einen", "und", "oder", "aber", "nicht", "kein", "keine", "ist", "sind", "war",
    "waren", "gewesen", "im", "in", "an", "auf", "mit", "bei", "von", "zu", "zur",
    "zum", "für", "über", "unter", "vor", "nach", "ich", "du", "er", "sie", "es", "wir",
    "ihr", "mich", "dich", "sich", "mir", "dir", "uns", "euch",
];

const ITALIAN_STOPWORDS: &[&str] = &[
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "del", "della", "dei",
    "degli", "delle", "in", "con", "su", "per", "tra", "fra", "e", "o", "ma", "non",
    "che", "chi", "cui", "come", "più", "meno", "molto", "poco", "tanto", "quanto",
    "tutto", "tutti", "tutte", "ogni", "alcuno", "alcuna", "alcuni", "alcune", "questo",
    "questa", "questi", "queste", "quello", "quella", "quelli", "quelle",
];

const PORTUGUESE_STOPWORDS: &[&str] = &[
    "o", "a", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "para", "com", "sem", "que", "como", "mais",
    "menos", "mas", "não", "sim", "se", "ou", "e", "este", "esta", "estes", "estas",
    "esse", "essa", "esses", "essas", "aquele", "aquela", "aqueles", "aquelas", "muito",
    "muita", "muitos", "muitas", "todo", "toda", "todos", "todas", "cada",
];

const DUTCH_STOPWORDS: &[&str] = &[
    "de", "het", "een", "en", "van", "in", "te", "is", "die", "niet", "met", "voor",
    "op", "aan", "ook", "dat", "maar", "zijn", "als", "wij", "jij", "hij", "zij", "het",
    "dit", "deze", "die", "wat", "wie", "waar", "wanneer", "hoe", "waarom", "alle",
    "sommige", "elke", "geen", "wel", "ja", "nee",
];

const RUSSIAN_STOPWORDS: &[&str] = &[
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все",
    "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по",
    "только", "ее", "мне", "было", "вот", "от", "меня", "о", "из", "ему", "теперь",
    "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или", "быть", "был", "него",
    "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего",
    "ей", "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_stopwords_case_insensitive() {
        assert!(is_stopword("the", Language::English));
        assert!(is_stopword("THE", Language::English));
        assert!(is_stopword("The", Language::English));
        assert!(!is_stopword("hello", Language::English));
    }

    #[test]
    fn spanish_stopwords() {
        assert!(is_stopword("el", Language::Spanish));
        assert!(is_stopword("LA", Language::Spanish));
        assert!(!is_stopword("hola", Language::Spanish));
    }

    #[test]
    fn french_stopwords() {
        assert!(is_stopword("le", Language::French));
        assert!(is_stopword("LES", Language::French));
        assert!(!is_stopword("bonjour", Language::French));
    }

    #[test]
    fn german_stopwords() {
        assert!(is_stopword("der", Language::German));
        assert!(is_stopword("DAS", Language::German));
        assert!(!is_stopword("haus", Language::German));
    }

    #[test]
    fn russian_stopwords() {
        assert!(is_stopword("и", Language::Russian));
        assert!(is_stopword("В", Language::Russian));
        assert!(!is_stopword("дом", Language::Russian));
    }

    #[test]
    fn list_is_sorted() {
        let list = stopword_list(Language::English);
        assert!(list.windows(2).all(|w| w[0] <= w[1]));
    }
}
