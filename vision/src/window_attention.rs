//! Window attention (EfficientViT 2026).
#![allow(missing_docs)]
use ndarray::{Array1, Array2, Array3, s};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WindowAttentionError { #[error("grid {h}x{w} not divisible by {ws}")] InvalidGrid { h: usize, w: usize, ws: usize }, #[error("dim mismatch: expected {expected}, got {actual}")] DimMismatch { expected: String, actual: String } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowAttentionConfig { pub heads: usize, pub head_dim: usize, pub window_size: usize, pub shift: bool, pub attn_dropout: f32, pub use_soft_mask: bool }
impl Default for WindowAttentionConfig { fn default() -> Self { Self { heads: 4, head_dim: 32, window_size: 7, shift: false, attn_dropout: 0.0, use_soft_mask: true } } }

pub struct WindowAttention { cfg: WindowAttentionConfig, w_q: Array2<f32>, w_k: Array2<f32>, w_v: Array2<f32>, w_o: Array2<f32>, soft_mask: Array1<f32> }

impl WindowAttention {
    pub fn new(channels: usize, cfg: WindowAttentionConfig) -> Result<Self, WindowAttentionError> {
        let inner = cfg.heads * cfg.head_dim;
        if inner != channels { return Err(WindowAttentionError::DimMismatch { expected: format!("channels ({channels}) == heads*head_dim ({inner})"), actual: format!("channels={channels}, heads={}, head_dim={}", cfg.heads, cfg.head_dim) }); }
        let scale = (channels as f32).sqrt().recip();
        Ok(Self { cfg, w_q: xavier(channels, channels, scale), w_k: xavier(channels, channels, scale), w_v: xavier(channels, channels, scale), w_o: xavier(channels, channels, scale), soft_mask: Array1::zeros(0) })
    }
    pub fn forward(&mut self, x: &Array3<f32>) -> Result<Array3<f32>, WindowAttentionError> {
        let (h, w, c) = x.dim(); let ws = self.cfg.window_size;
        if h % ws != 0 || w % ws != 0 { return Err(WindowAttentionError::InvalidGrid { h, w, ws }); }
        let shifted = if self.cfg.shift { roll_2d(x, ws / 2) } else { x.clone() };
        let nh = h / ws; let nw = w / ws; let mut out = Array3::zeros((h, w, c));
        for ih in 0..nh { for iw in 0..nw {
            let mut patches = Array2::zeros((ws * ws, c));
            for dy in 0..ws { for dx in 0..ws { let py = ih*ws+dy; let px = iw*ws+dx; patches.row_mut(dy*ws+dx).assign(&shifted.slice(s![py, px, ..])); } }
            let q = patches.dot(&self.w_q.t()); let k = patches.dot(&self.w_k.t()); let v = patches.dot(&self.w_v.t());
            let mut window_out = Array2::zeros((ws*ws, c));
            for hi in 0..self.cfg.heads { let hd = self.cfg.head_dim; let qh = q.slice(s![.., hi*hd..(hi+1)*hd]).to_owned(); let kh = k.slice(s![.., hi*hd..(hi+1)*hd]).to_owned(); let vh = v.slice(s![.., hi*hd..(hi+1)*hd]).to_owned();
                let scale = (hd as f32).sqrt().recip(); let mut attn = qh.dot(&kh.t()); attn.mapv_inplace(|a| a * scale);
                if self.cfg.use_soft_mask && self.soft_mask.len() == ws*ws { for r in 0..ws*ws { let m = 1.0_f32 / (1.0 + (-self.soft_mask[r]).exp()); attn.row_mut(r).mapv_inplace(|a| a * m); } }
                softmax_rows(&mut attn);
                let head_out = attn.dot(&vh);
                for r in 0..ws*ws { for cc in 0..hd { window_out[(r, hi*hd+cc)] = head_out[(r, cc)]; } }
            }
            let projected = window_out.dot(&self.w_o.t());
            for dy in 0..ws { for dx in 0..ws { let py = ih*ws+dy; let px = iw*ws+dx; out.slice_mut(s![py, px, ..]).assign(&projected.row(dy*ws+dx)); } }
        }}
        if self.cfg.shift { out = roll_2d(&out, ws.wrapping_sub(ws / 2)); }
        Ok(out)
    }
    pub fn set_soft_mask(&mut self, s: Array1<f32>) { self.soft_mask = s; }
}

fn xavier(fi: usize, fo: usize, sc: f32) -> Array2<f32> { let lim = (6.0_f32 / (fi + fo) as f32).sqrt() * sc; Array2::from_shape_fn((fo, fi), |(i, j)| { let v = ((i + j) as f32 + 1.0) * lim / (fi.max(fo) as f32); v.tanh() * lim }) }
fn softmax_rows(a: &mut Array2<f32>) { for mut row in a.rows_mut() { let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max); let mut s = 0.0; for v in row.iter_mut() { *v = (*v - mx).exp(); s += *v; } if s > 0.0 { for v in row.iter_mut() { *v /= s; } } } }
fn roll_2d(x: &Array3<f32>, dh: usize) -> Array3<f32> { let (h, w, c) = x.dim(); let dh = dh % h; if dh == 0 { return x.clone(); } let mut o = Array3::zeros((h, w, c)); for i in 0..h { let src = (i + h - dh) % h; o.slice_mut(s![i, .., ..]).assign(&x.slice(s![src, .., ..])); } o }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn forward_shape() { let mut a = WindowAttention::new(64, WindowAttentionConfig::default()).unwrap(); let x = Array3::zeros((14, 14, 64)); let y = a.forward(&x).unwrap(); assert_eq!(y.dim(), (14, 14, 64)); }
    #[test] fn invalid_grid() { let mut a = WindowAttention::new(64, WindowAttentionConfig::default()).unwrap(); let x = Array3::zeros((15, 14, 64)); assert!(matches!(a.forward(&x), Err(WindowAttentionError::InvalidGrid { .. }))); }
    #[test] fn shifted() { let cfg = WindowAttentionConfig { shift: true, ..Default::default() }; let mut a = WindowAttention::new(64, cfg).unwrap(); let x = Array3::from_shape_fn((14, 14, 64), |(i, j, _)| (i + j) as f32); let y = a.forward(&x).unwrap(); assert_eq!(y.dim(), (14, 14, 64)); }
    #[test] fn soft_mask() { let mut a = WindowAttention::new(64, WindowAttentionConfig::default()).unwrap(); a.set_soft_mask(Array1::from(vec![0.0; 49])); let x = Array3::zeros((14, 14, 64)); let y = a.forward(&x).unwrap(); assert_eq!(y.dim(), (14, 14, 64)); }
}
