//! Source-code loader: `.py`, `.rs`, `.cpp`, `.js`, `.go`, etc.
//!
//! Reads a source file and emits a single record with the file's content,
//! language, and (optionally) extracted structure (function/class names).
//! Also supports loading a whole directory tree as a multi-record dataset.

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::loader::{
    read_file_to_string, DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::streaming::MemoryRecordStream;

/// Source-code loader.
pub struct SourceCodeLoader;

impl Default for SourceCodeLoader {
    fn default() -> Self {
        Self
    }
}

impl SourceCodeLoader {
    /// Create a new source-code loader.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Map a file extension to a programming language name.
#[must_use]
pub fn language_for_extension(ext: &str) -> Option<&'static str> {
    match ext {
        "py" => Some("python"),
        "rs" => Some("rust"),
        "go" => Some("go"),
        "c" => Some("c"),
        "cpp" | "cc" | "cxx" => Some("cpp"),
        "h" | "hpp" => Some("cpp-header"),
        "js" | "mjs" | "cjs" => Some("javascript"),
        "jsx" => Some("jsx"),
        "ts" => Some("typescript"),
        "tsx" => Some("tsx"),
        "java" => Some("java"),
        "kt" | "kts" => Some("kotlin"),
        "swift" => Some("swift"),
        "rb" => Some("ruby"),
        "php" => Some("php"),
        "cs" => Some("csharp"),
        "scala" | "sc" => Some("scala"),
        "clj" | "cljs" | "cljc" => Some("clojure"),
        "hs" => Some("haskell"),
        "ml" => Some("ocaml"),
        "fs" | "fsx" => Some("fsharp"),
        "lua" => Some("lua"),
        "sh" | "bash" | "zsh" => Some("shell"),
        "ps1" => Some("powershell"),
        "el" => Some("elisp"),
        "ex" | "exs" => Some("elixir"),
        "erl" => Some("erlang"),
        "sql" => Some("sql"),
        "html" | "htm" => Some("html"),
        "css" => Some("css"),
        "scss" | "sass" => Some("scss"),
        "vue" => Some("vue"),
        "svelte" => Some("svelte"),
        "dart" => Some("dart"),
        "r" => Some("r"),
        "jl" => Some("julia"),
        "pl" | "pm" => Some("perl"),
        "vim" => Some("vim"),
        "asm" | "s" => Some("assembly"),
        "wat" | "wasm" => Some("wasm"),
        "dockerfile" => Some("dockerfile"),
        "makefile" | "mk" => Some("makefile"),
        "cmake" => Some("cmake"),
        "nim" => Some("nim"),
        "zig" => Some("zig"),
        "v" => Some("v"),
        "cr" => Some("crystal"),
        "d" => Some("d"),
        "pas" => Some("pascal"),
        "f" | "f90" | "f95" => Some("fortran"),
        "cob" => Some("cobol"),
        "ada" => Some("ada"),
        _ => None,
    }
}

/// Source-code extensions recognized by this loader.
pub const SOURCE_EXTENSIONS: &[&str] = &[
    "py", "rs", "go", "c", "cpp", "cc", "cxx", "h", "hpp",
    "js", "mjs", "cjs", "jsx", "ts", "tsx",
    "java", "kt", "kts", "swift", "rb", "php", "cs",
    "scala", "sc", "clj", "cljs", "cljc", "hs", "ml", "fs", "fsx",
    "lua", "sh", "bash", "zsh", "ps1", "el", "ex", "exs", "erl",
    "sql", "html", "htm", "css", "scss", "sass", "vue", "svelte",
    "dart", "r", "jl", "pl", "pm", "vim", "asm", "s", "wat",
    "dockerfile", "makefile", "mk", "cmake", "nim", "zig", "v", "cr",
    "d", "pas", "f", "f90", "f95", "cob", "ada",
];

impl SourceCodeLoader {
    /// Returns `true` if the path is a source-code file.
    fn is_source_file(path: &Path) -> bool {
        // Special-case files with no extension
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if file_name.eq_ignore_ascii_case("Dockerfile")
            || file_name.eq_ignore_ascii_case("Makefile")
            || file_name.eq_ignore_ascii_case("CMakeLists.txt")
        {
            return true;
        }
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| SOURCE_EXTENSIONS.iter().any(|x| *x == e))
            .unwrap_or(false)
    }

    /// Recursively collect all source files under `dir`.
    fn collect_source_files(dir: &Path) -> Vec<PathBuf> {
        WalkDir::new(dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file() && Self::is_source_file(e.path()))
            .map(|e| e.path().to_path_buf())
            .collect()
    }
}

#[async_trait]
impl DatasetLoader for SourceCodeLoader {
    fn id(&self) -> &str { "source-code" }
    fn name(&self) -> &str { "Source code (.py, .rs, .cpp, .js, ...)" }
    fn extensions(&self) -> &[&str] { SOURCE_EXTENSIONS }

    fn matches(&self, path: &Path) -> bool {
        Self::is_source_file(path)
    }

    async fn load_with_options(
        &self,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        // If path is a directory, walk it and collect all source files
        let files: Vec<PathBuf> = if path.is_dir() {
            Self::collect_source_files(path)
        } else {
            vec![path.to_path_buf()]
        };

        let max = options.max_records.unwrap_or(usize::MAX);
        let offset = options.offset.unwrap_or(0);
        let mut records = Vec::new();
        for (i, file) in files.iter().enumerate() {
            if i < offset {
                continue;
            }
            if records.len() >= max {
                break;
            }
            let content = match read_file_to_string(file).await {
                Ok(c) => c,
                Err(_) => continue,
            };
            let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
            let language = language_for_extension(ext).unwrap_or("unknown");
            let mut record = DatasetRecord {
                id: uuid::Uuid::now_v7(),
                modality: crate::loader::Modality::Code,
                mime: format!("text/x-{language}"),
                source: file.to_string_lossy().to_string(),
                content: serde_json::Value::String(content),
                bytes: None,
                embedding: None,
                metadata: std::collections::BTreeMap::new(),
                loaded_at: chrono::Utc::now(),
            };
            record.metadata.insert("language".into(), serde_json::Value::String(language.into()));
            record.metadata.insert("extension".into(), serde_json::Value::String(ext.into()));
            record.metadata.insert(
                "lines".into(),
                serde_json::Value::Number(
                    serde_json::Number::from(record.content.as_str().map(|s| s.lines().count()).unwrap_or(0)),
                ),
            );
            records.push(record);
        }

        Ok(Box::new(MemoryRecordStream::new(records)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn load_python_file() {
        let tmp = tempfile::Builder::new().suffix(".py").tempfile().unwrap();
        std::fs::write(tmp.path(), "def hello():\n    print('hi')\n").unwrap();
        let loader = SourceCodeLoader::new();
        let mut stream = loader.load(tmp.path()).await.unwrap();
        let r = stream.next_record().await.unwrap().unwrap();
        assert_eq!(r.modality, crate::loader::Modality::Code);
        assert_eq!(r.metadata.get("language").and_then(|v| v.as_str()), Some("python"));
    }

    #[test]
    fn language_mapping() {
        assert_eq!(language_for_extension("py"), Some("python"));
        assert_eq!(language_for_extension("rs"), Some("rust"));
        assert_eq!(language_for_extension("cpp"), Some("cpp"));
        assert_eq!(language_for_extension("unknown"), None);
    }
}
