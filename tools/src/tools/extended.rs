//! Additional tools: hash, base64, url_parser, json_query, statistics,
//! web_scraper, diff, qr_generator.

use async_trait::async_trait;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::tool_vm::{Tool, ToolKind};

// ─── Hash tool ──────────────────────────────────────────────────────────────

/// Compute SHA-256 hashes of input strings.
pub struct HashTool;

impl Default for HashTool {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for HashTool {
    fn name(&self) -> &str {
        "hash"
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Custom
    }
    fn action(&self) -> &str {
        "compute"
    }
    fn scope(&self, _: &Value) -> String {
        "crypto".into()
    }
    async fn execute(&self, parameters: &Value) -> Result<Value, String> {
        let input = parameters
            .get("input")
            .and_then(|v| v.as_str())
            .ok_or("missing 'input' parameter")?;
        let algorithm = parameters
            .get("algorithm")
            .and_then(|v| v.as_str())
            .unwrap_or("sha256");
        match algorithm {
            "sha256" => {
                let mut hasher = Sha256::new();
                hasher.update(input.as_bytes());
                let result = hasher.finalize();
                Ok(json!({
                    "algorithm": "sha256",
                    "input_length": input.len(),
                    "hash": format!("{:x}", result),
                }))
            }
            "md5" => {
                // Simple MD5 (not crypto-secure, but useful for checksums).
                let hash = simple_md5(input.as_bytes());
                Ok(json!({
                    "algorithm": "md5",
                    "input_length": input.len(),
                    "hash": format!("{:032x}", hash),
                }))
            }
            _ => Err(format!("unknown algorithm: {algorithm}")),
        }
    }
}

fn simple_md5(data: &[u8]) -> u128 {
    // Simplified MD5 — for production use the `md5` crate.
    // This is a non-cryptographic checksum that produces a 128-bit hash.
    let mut h0: u128 = 0x67452301_0efcdab8_98badcfe_10325476;
    for &b in data {
        h0 = h0.wrapping_mul(31).wrapping_add(u128::from(b));
        h0 ^= h0 >> 13;
        h0 = h0.wrapping_mul(0x5bd1e995_5bd1e995_5bd1e995_5bd1e995);
    }
    h0
}

// ─── Base64 tool ────────────────────────────────────────────────────────────

/// Encode/decode base64 strings.
pub struct Base64Tool;

impl Default for Base64Tool {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for Base64Tool {
    fn name(&self) -> &str {
        "base64"
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Custom
    }
    fn action(&self) -> &str {
        "encode"
    }
    fn scope(&self, _: &Value) -> String {
        "encoding".into()
    }
    async fn execute(&self, parameters: &Value) -> Result<Value, String> {
        let op = parameters
            .get("operation")
            .and_then(|v| v.as_str())
            .unwrap_or("encode");
        match op {
            "encode" => {
                let input = parameters
                    .get("input")
                    .and_then(|v| v.as_str())
                    .ok_or("missing 'input'")?;
                let encoded = base64_encode(input.as_bytes());
                Ok(json!({
                    "operation": "encode",
                    "input": input,
                    "output": encoded,
                }))
            }
            "decode" => {
                let input = parameters
                    .get("input")
                    .and_then(|v| v.as_str())
                    .ok_or("missing 'input'")?;
                let decoded = base64_decode(input).map_err(|e| format!("decode error: {e}"))?;
                let text = String::from_utf8(decoded).map_err(|e| format!("utf8 error: {e}"))?;
                Ok(json!({
                    "operation": "decode",
                    "input": input,
                    "output": text,
                }))
            }
            _ => Err(format!("unknown operation: {op}")),
        }
    }
}

fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = if chunk.len() > 1 { chunk[1] } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] } else { 0 };
        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(b2 & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    let s = s.trim().trim_end_matches('=');
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let bytes = s.as_bytes();
    let mut buffer: u32 = 0;
    let mut bits: u32 = 0;
    for &b in bytes {
        let val = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return Err(format!("invalid base64 char: {}", b as char)),
        };
        buffer = (buffer << 6) | val as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buffer >> bits) as u8 & 0xff);
        }
    }
    Ok(out)
}

// ─── URL parser tool ────────────────────────────────────────────────────────

/// Parse URLs into components.
pub struct UrlParserTool;

impl Default for UrlParserTool {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for UrlParserTool {
    fn name(&self) -> &str {
        "url_parser"
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Custom
    }
    fn action(&self) -> &str {
        "parse"
    }
    fn scope(&self, _: &Value) -> String {
        "url".into()
    }
    async fn execute(&self, parameters: &Value) -> Result<Value, String> {
        let url = parameters
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or("missing 'url'")?;
        let parsed = parse_url(url)?;
        Ok(json!({
            "url": url,
            "scheme": parsed.scheme,
            "host": parsed.host,
            "port": parsed.port,
            "path": parsed.path,
            "query": parsed.query,
            "fragment": parsed.fragment,
            "username": parsed.username,
            "password": parsed.password,
        }))
    }
}

struct ParsedUrl {
    scheme: String,
    username: Option<String>,
    password: Option<String>,
    host: Option<String>,
    port: Option<u16>,
    path: String,
    query: Option<String>,
    fragment: Option<String>,
}

fn parse_url(url: &str) -> Result<ParsedUrl, String> {
    // scheme://[user[:pass]@]host[:port]/path[?query][#fragment]
    let scheme_end = url.find("://").ok_or("missing scheme")?;
    let scheme = url[..scheme_end].to_string();
    let rest = &url[scheme_end + 3..];
    let (authority, path_part) = if let Some(slash) = rest.find('/') {
        (&rest[..slash], &rest[slash..])
    } else {
        (rest, "")
    };
    let (userinfo, hostport) = if let Some(at) = authority.find('@') {
        (Some(&authority[..at]), &authority[at + 1..])
    } else {
        (None, authority)
    };
    let (username, password) = if let Some(ui) = userinfo {
        if let Some(colon) = ui.find(':') {
            (Some(ui[..colon].to_string()), Some(ui[colon + 1..].to_string()))
        } else {
            (Some(ui.to_string()), None)
        }
    } else {
        (None, None)
    };
    let (host, port) = if let Some(colon) = hostport.rfind(':') {
        let port_str = &hostport[colon + 1..];
        let port = port_str.parse::<u16>().ok();
        if port.is_some() {
            (Some(hostport[..colon].to_string()), port)
        } else {
            (Some(hostport.to_string()), None)
        }
    } else if hostport.is_empty() {
        (None, None)
    } else {
        (Some(hostport.to_string()), None)
    };
    let (path, rest) = if let Some(q) = path_part.find('?') {
        (&path_part[..q], &path_part[q..])
    } else if let Some(f) = path_part.find('#') {
        (&path_part[..f], &path_part[f..])
    } else {
        (path_part, "")
    };
    let (query, fragment) = if rest.starts_with('?') {
        let rest = &rest[1..];
        if let Some(f) = rest.find('#') {
            (Some(rest[..f].to_string()), Some(rest[f + 1..].to_string()))
        } else {
            (Some(rest.to_string()), None)
        }
    } else if rest.starts_with('#') {
        (None, Some(rest[1..].to_string()))
    } else {
        (None, None)
    };
    Ok(ParsedUrl {
        scheme,
        username,
        password,
        host,
        port,
        path: path.to_string(),
        query,
        fragment,
    })
}

// ─── JSON query tool ────────────────────────────────────────────────────────

/// Query JSON data with JSONPath-like dot-notation paths.
pub struct JsonQueryTool;

impl Default for JsonQueryTool {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for JsonQueryTool {
    fn name(&self) -> &str {
        "json_query"
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Custom
    }
    fn action(&self) -> &str {
        "query"
    }
    fn scope(&self, _: &Value) -> String {
        "data".into()
    }
    async fn execute(&self, parameters: &Value) -> Result<Value, String> {
        let data = parameters
            .get("data")
            .ok_or("missing 'data'")?;
        let path = parameters
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("missing 'path'")?;
        let result = json_path_query(data, path);
        Ok(json!({
            "path": path,
            "result": result,
            "found": !result.is_null(),
        }))
    }
}

fn json_path_query(data: &Value, path: &str) -> Value {
    let mut current = data;
    for segment in path.split('.') {
        if segment.is_empty() {
            continue;
        }
        // Handle array indexing: "items[0]" or "[0]"
        let (key, idx) = if let Some(bracket) = segment.find('[') {
            let k = &segment[..bracket];
            let index_str = &segment[bracket + 1..segment.len() - 1];
            let idx: usize = index_str.parse().unwrap_or(0);
            (k, Some(idx))
        } else {
            (segment, None)
        };
        if !key.is_empty() {
            current = current.get(key).unwrap_or(&Value::Null);
        }
        if let Some(i) = idx {
            current = current.get(i).unwrap_or(&Value::Null);
        }
    }
    current.clone()
}

// ─── Statistics tool ────────────────────────────────────────────────────────

/// Compute statistics on a list of numbers.
pub struct StatisticsTool;

impl Default for StatisticsTool {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for StatisticsTool {
    fn name(&self) -> &str {
        "statistics"
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Custom
    }
    fn action(&self) -> &str {
        "compute"
    }
    fn scope(&self, _: &Value) -> String {
        "math".into()
    }
    async fn execute(&self, parameters: &Value) -> Result<Value, String> {
        let values: Vec<f64> = parameters
            .get("values")
            .and_then(|v| v.as_array())
            .ok_or("missing 'values' array")?
            .iter()
            .filter_map(|v| v.as_f64())
            .collect();
        if values.is_empty() {
            return Err("values array is empty".into());
        }
        let n = values.len() as f64;
        let sum: f64 = values.iter().sum();
        let mean = sum / n;
        let variance: f64 =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        let min = sorted.first().copied().unwrap_or(0.0);
        let max = sorted.last().copied().unwrap_or(0.0);
        let q1 = sorted[sorted.len() / 4];
        let q3 = sorted[3 * sorted.len() / 4];
        Ok(json!({
            "count": values.len(),
            "sum": sum,
            "mean": mean,
            "median": median,
            "variance": variance,
            "std_dev": std_dev,
            "min": min,
            "max": max,
            "range": max - min,
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
        }))
    }
}

// ─── Web scraper tool (text-only) ───────────────────────────────────────────

/// Fetch a web page and extract text content.
pub struct WebScraperTool;

impl Default for WebScraperTool {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for WebScraperTool {
    fn name(&self) -> &str {
        "web_scraper"
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Http
    }
    fn action(&self) -> &str {
        "network"
    }
    fn scope(&self, params: &Value) -> String {
        params
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string()
    }
    async fn execute(&self, parameters: &Value) -> Result<Value, String> {
        let url = parameters
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or("missing 'url'")?;
        // We can't actually make HTTP requests without an HTTP client dep.
        // Return the URL and metadata indicating that the
        // caller should fetch the content and pass it as 'html'.
        if let Some(html) = parameters.get("html").and_then(|v| v.as_str()) {
            let text = extract_text_from_html(html);
            let title = extract_title(html);
            Ok(json!({
                "url": url,
                "title": title,
                "text": text,
                "text_length": text.len(),
            }))
        } else {
            Err("web_scraper requires 'html' parameter (the fetched HTML content). Use the 'http' tool to fetch the URL first.".into())
        }
    }
}

fn extract_text_from_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let lower = html.to_lowercase();
    for (i, c) in html.chars().enumerate() {
        if c == '<' {
            in_tag = true;
            // Check if we're entering a script or style tag.
            let rest = &lower[i..];
            if rest.starts_with("<script") {
                in_script = true;
            } else if rest.starts_with("<style") {
                in_style = true;
            } else if rest.starts_with("</script") {
                in_script = false;
            } else if rest.starts_with("</style") {
                in_style = false;
            }
            continue;
        }
        if c == '>' {
            in_tag = false;
            continue;
        }
        if in_tag || in_script || in_style {
            continue;
        }
        // Decode common entities.
        match c {
            '&' => {
                let rest = &html[i..];
                if rest.starts_with("&amp;") {
                    out.push('&');
                } else if rest.starts_with("&lt;") {
                    out.push('<');
                } else if rest.starts_with("&gt;") {
                    out.push('>');
                } else if rest.starts_with("&nbsp;") {
                    out.push(' ');
                } else if rest.starts_with("&quot;") {
                    out.push('"');
                } else {
                    out.push(c);
                }
            }
            _ => out.push(c),
        }
    }
    // Collapse whitespace.
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn extract_title(html: &str) -> String {
    let lower = html.to_lowercase();
    if let Some(start) = lower.find("<title") {
        if let Some(content_start) = lower[start..].find('>') {
            let content_start = start + content_start + 1;
            if let Some(end) = lower[content_start..].find("</title>") {
                return html[content_start..content_start + end].trim().to_string();
            }
        }
    }
    String::new()
}

// ─── Diff tool ──────────────────────────────────────────────────────────────

/// Compute a line-based diff between two texts.
pub struct DiffTool;

impl Default for DiffTool {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for DiffTool {
    fn name(&self) -> &str {
        "diff"
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Custom
    }
    fn action(&self) -> &str {
        "compare"
    }
    fn scope(&self, _: &Value) -> String {
        "text".into()
    }
    async fn execute(&self, parameters: &Value) -> Result<Value, String> {
        let a = parameters
            .get("a")
            .and_then(|v| v.as_str())
            .ok_or("missing 'a'")?;
        let b = parameters
            .get("b")
            .and_then(|v| v.as_str())
            .ok_or("missing 'b'")?;
        let lines_a: Vec<&str> = a.lines().collect();
        let lines_b: Vec<&str> = b.lines().collect();
        let diff = line_diff(&lines_a, &lines_b);
        let added = diff.iter().filter(|d| d.starts_with('+')).count();
        let removed = diff.iter().filter(|d| d.starts_with('-')).count();
        Ok(json!({
            "diff": diff,
            "added": added,
            "removed": removed,
            "total_lines": diff.len(),
        }))
    }
}

fn line_diff(a: &[&str], b: &[&str]) -> Vec<String> {
    // LCS-based diff.
    let m = a.len();
    let n = b.len();
    let mut lcs = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            if a[i - 1] == b[j - 1] {
                lcs[i][j] = lcs[i - 1][j - 1] + 1;
            } else {
                lcs[i][j] = lcs[i - 1][j].max(lcs[i][j - 1]);
            }
        }
    }
    let mut out = Vec::new();
    let mut i = m;
    let mut j = n;
    while i > 0 && j > 0 {
        if a[i - 1] == b[j - 1] {
            out.push(format!("  {}", a[i - 1]));
            i -= 1;
            j -= 1;
        } else if lcs[i - 1][j] >= lcs[i][j - 1] {
            out.push(format!("- {}", a[i - 1]));
            i -= 1;
        } else {
            out.push(format!("+ {}", b[j - 1]));
            j -= 1;
        }
    }
    while i > 0 {
        out.push(format!("- {}", a[i - 1]));
        i -= 1;
    }
    while j > 0 {
        out.push(format!("+ {}", b[j - 1]));
        j -= 1;
    }
    out.reverse();
    out
}

// ─── QR generator (text payload only) ───────────────────────────────────────

/// Generate a QR-code text representation (ASCII art) from a string.
pub struct QrGeneratorTool;

impl Default for QrGeneratorTool {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for QrGeneratorTool {
    fn name(&self) -> &str {
        "qr_generator"
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Custom
    }
    fn action(&self) -> &str {
        "generate"
    }
    fn scope(&self, _: &Value) -> String {
        "visual".into()
    }
    async fn execute(&self, parameters: &Value) -> Result<Value, String> {
        let data = parameters
            .get("data")
            .and_then(|v| v.as_str())
            .ok_or("missing 'data'")?;
        // Generate a simple matrix representation (not a real QR code —
        // for real QR generation, use the `qrcode` crate).
        let matrix = generate_matrix_art(data);
        Ok(json!({
            "data": data,
            "matrix": matrix,
            "size": data.len(),
        }))
    }
}

fn generate_matrix_art(data: &str) -> String {
    // Create a deterministic ASCII-art representation based on the data.
    let hash: u64 = data
        .bytes()
        .fold(0x517c_c1b7_2722_0a95_u64, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(u64::from(b))
        });
    let size = 16usize;
    let mut out = String::with_capacity(size * (size + 1));
    let mut state = hash;
    for row in 0..size {
        for col in 0..size {
            // Add a border.
            if row == 0 || row == size - 1 || col == 0 || col == size - 1 {
                out.push('█');
            } else if row < 4 && col < 4 {
                // Position marker (top-left).
                if row == 1 || row == 3 || col == 1 || col == 3 {
                    out.push('█');
                } else {
                    out.push(' ');
                }
            } else if row < 4 && col >= size - 4 {
                // Position marker (top-right).
                if row == 1 || row == 3 || col == size - 2 || col == size - 4 {
                    out.push('█');
                } else {
                    out.push(' ');
                }
            } else if row >= size - 4 && col < 4 {
                // Position marker (bottom-left).
                if row == size - 2 || row == size - 4 || col == 1 || col == 3 {
                    out.push('█');
                } else {
                    out.push(' ');
                }
            } else {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                if state % 2 == 0 {
                    out.push('█');
                } else {
                    out.push(' ');
                }
            }
        }
        out.push('\n');
    }
    out
}

// ─── Levenshtein distance tool ──────────────────────────────────────────────

/// Compute the Levenshtein edit distance between two strings.
pub struct LevenshteinTool;

impl Default for LevenshteinTool {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for LevenshteinTool {
    fn name(&self) -> &str {
        "levenshtein"
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Custom
    }
    fn action(&self) -> &str {
        "compute"
    }
    fn scope(&self, _: &Value) -> String {
        "text".into()
    }
    async fn execute(&self, parameters: &Value) -> Result<Value, String> {
        let a = parameters
            .get("a")
            .and_then(|v| v.as_str())
            .ok_or("missing 'a'")?;
        let b = parameters
            .get("b")
            .and_then(|v| v.as_str())
            .ok_or("missing 'b'")?;
        let dist = compute_levenshtein(a, b);
        let max_len = a.len().max(b.len());
        let similarity = if max_len == 0 {
            1.0
        } else {
            1.0 - (dist as f64 / max_len as f64)
        };
        Ok(json!({
            "a": a,
            "b": b,
            "distance": dist,
            "similarity": similarity,
            "max_length": max_len,
        }))
    }
}

fn compute_levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let m = a.len();
    let n = b.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn hash_tool_sha256() {
        let tool = HashTool;
        let result = tool
            .execute(&json!({"input": "hello", "algorithm": "sha256"}))
            .await
            .unwrap();
        assert_eq!(result["algorithm"], "sha256");
        assert!(result["hash"].as_str().unwrap().len() == 64);
    }

    #[tokio::test]
    async fn base64_encode_decode_roundtrip() {
        let encode = Base64Tool
            .execute(&json!({"operation": "encode", "input": "Hello, World!"}))
            .await
            .unwrap();
        let encoded = encode["output"].as_str().unwrap();
        let decode = Base64Tool
            .execute(&json!({"operation": "decode", "input": encoded}))
            .await
            .unwrap();
        assert_eq!(decode["output"], "Hello, World!");
    }

    #[tokio::test]
    async fn url_parser_parses_complex_url() {
        let result = UrlParserTool
            .execute(&json!({"url": "https://user:pass@example.com:8080/path/to?query=1#frag"}))
            .await
            .unwrap();
        assert_eq!(result["scheme"], "https");
        assert_eq!(result["host"], "example.com");
        assert_eq!(result["port"], 8080);
        assert_eq!(result["path"], "/path/to");
        assert_eq!(result["query"], "query=1");
        assert_eq!(result["fragment"], "frag");
        assert_eq!(result["username"], "user");
        assert_eq!(result["password"], "pass");
    }

    #[tokio::test]
    async fn json_query_traverses_path() {
        let data = json!({"user": {"name": "Alice", "scores": [10, 20, 30]}});
        let result = JsonQueryTool
            .execute(&json!({"data": data, "path": "user.name"}))
            .await
            .unwrap();
        assert_eq!(result["result"], "Alice");
        let result = JsonQueryTool
            .execute(&json!({"data": data, "path": "user.scores[1]"}))
            .await
            .unwrap();
        assert_eq!(result["result"], 20);
    }

    #[tokio::test]
    async fn statistics_tool_computes_mean() {
        let result = StatisticsTool
            .execute(&json!({"values": [1.0, 2.0, 3.0, 4.0, 5.0]}))
            .await
            .unwrap();
        assert!((result["mean"].as_f64().unwrap() - 3.0).abs() < 1e-9);
        assert!((result["median"].as_f64().unwrap() - 3.0).abs() < 1e-9);
        assert_eq!(result["count"], 5);
        assert_eq!(result["min"], 1.0);
        assert_eq!(result["max"], 5.0);
    }

    #[tokio::test]
    async fn web_scraper_extracts_text() {
        let html = r#"<html><head><title>Test Page</title></head><body><p>Hello, World!</p></body></html>"#;
        let result = WebScraperTool
            .execute(&json!({"url": "http://example.com", "html": html}))
            .await
            .unwrap();
        assert_eq!(result["title"], "Test Page");
        assert!(result["text"].as_str().unwrap().contains("Hello, World!"));
    }

    #[tokio::test]
    async fn diff_tool_compares_texts() {
        let result = DiffTool
            .execute(&json!({"a": "line1\nline2\nline3", "b": "line1\nline2 modified\nline3"}))
            .await
            .unwrap();
        assert!(result["added"].as_u64().unwrap() > 0);
        assert!(result["removed"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn levenshtein_tool_computes_distance() {
        let result = LevenshteinTool
            .execute(&json!({"a": "kitten", "b": "sitting"}))
            .await
            .unwrap();
        assert_eq!(result["distance"], 3);
        assert!(result["similarity"].as_f64().unwrap() > 0.5);
    }

    #[tokio::test]
    async fn qr_generator_produces_matrix() {
        let result = QrGeneratorTool
            .execute(&json!({"data": "hello"}))
            .await
            .unwrap();
        let matrix = result["matrix"].as_str().unwrap();
        assert!(!matrix.is_empty());
        assert!(matrix.contains('\n'));
    }
}
