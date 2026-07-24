//! Transliteration: convert text between scripts.

/// Transliterate Cyrillic text to Latin (scientific / ISO 9:1995 style).
#[must_use]
pub fn cyrillic_to_latin(text: &str) -> String {
    text.chars()
        .map(|c| {
            let code = c as u32;
            if (0x0400..=0x04FF).contains(&code) {
                cyrillic_to_latin_char(c)
            } else {
                c.to_string()
            }
        })
        .collect()
}

/// Transliterate Latin text to Cyrillic (Russian, GOST/Scholarly transliteration).
#[must_use]
pub fn latin_to_cyrillic(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let lower = chars[i].to_ascii_lowercase();
        let next_lower = chars.get(i + 1).map(|c| c.to_ascii_lowercase());
        // Handle digraphs first.
        let (transliterated, consumed) = match (lower, next_lower) {
            ('z', Some('h')) => ("ж", 2),
            ('k', Some('h')) => ("х", 2),
            ('t', Some('s')) => ("ц", 2),
            ('c', Some('h')) => ("ч", 2),
            ('s', Some('h')) => ("ш", 2),
            ('s', Some('\u{030c}')) => ("щ", 2),
            ('y', Some('a')) => ("я", 2),
            ('y', Some('u')) => ("ю", 2),
            ('y', Some('o')) => ("ё", 2),
            ('e', Some('\'')) => ("э", 2),
            _ => {
                let s = latin_to_cyrillic_char(lower);
                (s, 1)
            }
        };
        // Preserve case: if the original was uppercase, uppercase the first char.
        if chars[i].is_uppercase() {
            let mut chars_iter = transliterated.chars();
            if let Some(first) = chars_iter.next() {
                for uc in first.to_uppercase() {
                    out.push(uc);
                }
                for c in chars_iter {
                    out.push(c);
                }
            }
        } else {
            out.push_str(transliterated);
        }
        i += consumed;
    }
    out
}

/// A simple pinyin-to-Chinese hint table (very limited; for full pinyin
/// conversion a dictionary is required).
#[must_use]
pub fn pinyin_to_chinese_hint(pinyin: &str) -> Vec<&'static str> {
    let lower = pinyin.to_lowercase();
    let mut hints = Vec::new();
    for word in lower.split_whitespace() {
        if let Some(hint) = PINYIN_TABLE.iter().find(|(p, _)| *p == word).map(|(_, c)| *c) {
            hints.push(hint);
        }
    }
    hints
}

fn cyrillic_to_latin_char(c: char) -> String {
    match c {
        'а' => "a".into(),
        'б' => "b".into(),
        'в' => "v".into(),
        'г' => "g".into(),
        'д' => "d".into(),
        'е' => "e".into(),
        'ё' => "e".into(),
        'ж' => "zh".into(),
        'з' => "z".into(),
        'и' => "i".into(),
        'й' => "j".into(),
        'к' => "k".into(),
        'л' => "l".into(),
        'м' => "m".into(),
        'н' => "n".into(),
        'о' => "o".into(),
        'п' => "p".into(),
        'р' => "r".into(),
        'с' => "s".into(),
        'т' => "t".into(),
        'у' => "u".into(),
        'ф' => "f".into(),
        'х' => "h".into(),
        'ц' => "c".into(),
        'ч' => "ch".into(),
        'ш' => "sh".into(),
        'щ' => "shh".into(),
        'ъ' => "\"".into(),
        'ы' => "y".into(),
        'ь' => "'".into(),
        'э' => "e".into(),
        'ю' => "yu".into(),
        'я' => "ya".into(),
        'А' => "A".into(),
        'Б' => "B".into(),
        'В' => "V".into(),
        'Г' => "G".into(),
        'Д' => "D".into(),
        'Е' => "E".into(),
        'Ё' => "E".into(),
        'Ж' => "Zh".into(),
        'З' => "Z".into(),
        'И' => "I".into(),
        'Й' => "J".into(),
        'К' => "K".into(),
        'Л' => "L".into(),
        'М' => "M".into(),
        'Н' => "N".into(),
        'О' => "O".into(),
        'П' => "P".into(),
        'Р' => "R".into(),
        'С' => "S".into(),
        'Т' => "T".into(),
        'У' => "U".into(),
        'Ф' => "F".into(),
        'Х' => "H".into(),
        'Ц' => "C".into(),
        'Ч' => "Ch".into(),
        'Ш' => "Sh".into(),
        'Щ' => "Shh".into(),
        'Ъ' => "\"".into(),
        'Ы' => "Y".into(),
        'Ь' => "'".into(),
        'Э' => "E".into(),
        'Ю' => "Yu".into(),
        'Я' => "Ya".into(),
        _ => c.to_string(),
    }
}

fn latin_to_cyrillic_char(c: char) -> &'static str {
    match c {
        'a' => "а",
        'b' => "б",
        'v' => "в",
        'g' => "г",
        'd' => "д",
        'e' => "е",
        'z' => "з",
        'i' => "и",
        'j' => "й",
        'k' => "к",
        'l' => "л",
        'm' => "м",
        'n' => "н",
        'o' => "о",
        'p' => "п",
        'r' => "р",
        's' => "с",
        't' => "т",
        'u' => "у",
        'f' => "ф",
        'y' => "ы",
        // Preserve whitespace and other non-letter characters.
        ' ' => " ",
        _ => "",
    }
}

const PINYIN_TABLE: &[(&str, &str)] = &[
    ("ni", "你"),
    ("hao", "好"),
    ("ma", "吗"),
    ("shi", "是"),
    ("de", "的"),
    ("wo", "我"),
    ("ni", "你"),
    ("ta", "他"),
    ("women", "我们"),
    ("nimen", "你们"),
    ("tamen", "他们"),
    ("zhongguo", "中国"),
    ("shijie", "世界"),
    ("nihao", "你好"),
    ("xiexie", "谢谢"),
    ("zaijian", "再见"),
    ("ai", "爱"),
    ("hanyu", "汉语"),
    ("yingyu", "英语"),
    ("richang", "日常"),
    ("shenghuo", "生活"),
    ("gongzuo", "工作"),
    ("xuexi", "学习"),
    ("jintian", "今天"),
    ("mingtian", "明天"),
    ("zuotian", "昨天"),
    ("shijian", "时间"),
    ("yidian", "一点"),
    ("name", "那么"),
    ("zheme", "这么"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cyrillic_to_latin() {
        assert_eq!(cyrillic_to_latin("привет"), "privet");
        assert_eq!(cyrillic_to_latin("мир"), "mir");
        assert_eq!(cyrillic_to_latin("Привет"), "Privet");
    }

    #[test]
    fn test_cyrillic_to_latin_preserves_non_cyrillic() {
        assert_eq!(cyrillic_to_latin("hello мир"), "hello mir");
        assert_eq!(cyrillic_to_latin("123"), "123");
    }

    #[test]
    fn test_latin_to_cyrillic() {
        assert_eq!(latin_to_cyrillic("privet"), "привет");
        assert_eq!(latin_to_cyrillic("mir"), "мир");
        assert_eq!(latin_to_cyrillic("Privet"), "Привет");
    }

    #[test]
    fn test_latin_to_cyrillic_handles_digraphs() {
        assert_eq!(latin_to_cyrillic("zh"), "ж");
        assert_eq!(latin_to_cyrillic("ch"), "ч");
        assert_eq!(latin_to_cyrillic("sh"), "ш");
        assert_eq!(latin_to_cyrillic("ya"), "я");
    }

    #[test]
    fn test_pinyin_table_works() {
        let hints = pinyin_to_chinese_hint("ni hao");
        assert!(hints.contains(&"你"));
        assert!(hints.contains(&"好"));
    }

    #[test]
    fn test_round_trip_cyrillic() {
        let original = "привет мир";
        let latin = cyrillic_to_latin(original);
        let back = latin_to_cyrillic(&latin);
        assert_eq!(back, original);
    }
}
