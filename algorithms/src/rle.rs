//! Run-length encoding for lossless compression of repetitive sequences.

/// Encode a byte slice using run-length encoding.
#[must_use]
pub fn rle_encode(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len());
    let mut current = data[0];
    let mut count = 1u8;
    for &b in &data[1..] {
        if b == current && count < 255 {
            count += 1;
        } else {
            out.push(count);
            out.push(current);
            current = b;
            count = 1;
        }
    }
    out.push(count);
    out.push(current);
    out
}

/// Decode a run-length encoded byte slice.
///
/// # Errors
/// Returns an error if the input has an odd length.
pub fn rle_decode(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() % 2 != 0 {
        return Err("invalid RLE data: odd length".into());
    }
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let count = data[i] as usize;
        let byte = data[i + 1];
        for _ in 0..count {
            out.push(byte);
        }
        i += 2;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_compresses_repetitive_data() {
        let input = b"aaaaabbbbbbcccccccdddddddeeeee";
        let encoded = rle_encode(input);
        assert!(encoded.len() < input.len());
        assert_eq!(rle_decode(&encoded).unwrap(), input);
    }

    #[test]
    fn empty_input() {
        assert!(rle_encode(b"").is_empty());
        assert!(rle_decode(b"").unwrap().is_empty());
    }

    #[test]
    fn single_byte() {
        let encoded = rle_encode(b"a");
        assert_eq!(encoded, vec![1, b'a']);
        assert_eq!(rle_decode(&encoded).unwrap(), b"a");
    }

    #[test]
    fn alternating_bytes() {
        let input = b"abababab";
        let encoded = rle_encode(input);
        // RLE doesn't compress alternating data; it actually grows it.
        assert!(encoded.len() >= input.len());
        assert_eq!(rle_decode(&encoded).unwrap(), input);
    }

    #[test]
    fn decode_rejects_odd_length() {
        assert!(rle_decode(&[1, 2, 3]).is_err());
    }

    #[test]
    fn long_run_is_split_at_255() {
        let input = vec![b'x'; 300];
        let encoded = rle_encode(&input);
        // 300 = 255 + 45 → two runs.
        assert_eq!(encoded, vec![255, b'x', 45, b'x']);
        assert_eq!(rle_decode(&encoded).unwrap(), input);
    }
}
