# qbitos-vision

Image processor: HOG + color histogram + LBP features, OCR, object detection.

> Part of the [QbitOS](../README.md) project · [Architecture](../docs/ARCHITECTURE.md) · [API](../docs/API.md) · [Deployment](../docs/DEPLOYMENT.md)

## Overview

A from-scratch vision pipeline (no OpenCV dependency).

## Features

- **Feature extraction**: HOG (Histogram of Oriented Gradients), color histograms, LBP (Local Binary Patterns)
- **OCR**: Otsu thresholding + connected component analysis + template matching
- **Object detection**: Contour analysis + Non-Maximum Suppression (NMS)

## Usage

```rust
use qbitos_vision::VisionModule;
let vision = VisionModule::default_config(memory.embedder.clone());
let result = vision.perceive(&image_bytes, &[VisionOp::Features, VisionOp::Ocr, VisionOp::Detect]);
```
