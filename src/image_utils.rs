//   Raw pixels: u8 values in [0, 255] per channel (R, G, B)
//   Normalized: f64 values in [-1.0, 1.0] per channel
//
// Normalization formula:
//   x_norm = (pixel_value / 127.5) - 1.0
//
// Denormalization (to save image):
//   pixel_value = (x_norm + 1.0) * 127.5
//   then clamp to [0, 255] and cast to u8

use image::{DynamicImage, ImageBuffer, RgbImage};
use std::path::Path;

/// Represents an image as a flat Vec<f64> of normalized pixel values.
/// Layout: [R_00, G_00, B_00, R_01, G_01, B_01, ..., R_(h-1)(w-1), ...]
/// where (row, col) ordering, 3 values per pixel.
#[derive(Debug, Clone)]
pub struct NormalizedImage {
    // normalized pixel values in [-1.0, 1.0]
    pub data: Vec<f64>,  
    pub width: u32,
    pub height: u32,
    // always 3 for RGB
    pub channels: u32,   
}

impl NormalizedImage {
    //Total number of scalar values (width × height × channels)
    pub fn len(&self) -> usize {
        (self.width * self.height * self.channels) as usize
    }

    // Get a normalized pixel value at (row, col, channel)
    pub fn get(&self, row: u32, col: u32, channel: u32) -> f64 {
        let idx = ((row * self.width + col) * self.channels + channel) as usize;
        self.data[idx]
    }
}

/// Load an image from disk and normalize to [-1.0, 1.0]
///
/// # Process
/// 1. Open file using the `image` crate (auto-detects format)
/// 2. Convert to RGB8 (3 channels, 8 bits each)
/// 3. Normalize each u8 value to f64 in [-1.0, 1.0]
pub fn load_and_normalize(path: &Path) -> anyhow::Result<NormalizedImage> {
    // image::open auto-detects image formats
    let img: DynamicImage = image::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to open image at {:?}: {}", path, e))?;

    // Convert to RGB8: ensures we always have exactly 3 channels
    let rgb: RgbImage = img.into_rgb8();
    let (width, height) = rgb.dimensions();

    if width == 0 || height == 0 {
        return Err(anyhow::anyhow!("Image at {:?} has zero dimensions", path));
    } else {
         println!("Image loaded: {}×{} pixels ({} values total)", width, height, rgb.len());
    }

    // Normalize: u8 [0,255] → f64 [-1, 1]
    // Formula: (pixel / 127.5) - 1.0
    //   pixel=0   → -1.0 (black)
    //   pixel=127 → -0.004 (≈ 0, middle gray)
    //   pixel=255 → +1.0 (white)
    let data: Vec<f64> = rgb.as_raw()
        .iter()
        .map(|&pixel| (pixel as f64 / 127.5) - 1.0)
        .collect();

    Ok(NormalizedImage {
        data,
        width,
        height,
        channels: 3,
    })
}

// Save a normalized image back to disk as a standard image file.
//
// # Process
// 1. Denormalize: f64 [-1.0, 1.0] → f64 [0.0, 255.0]
// 2. Clamp to [0.0, 255.0] (important! diffusion can push values slightly outside range)
// 3. Cast to u8
// 4. Write to file
pub fn save_normalized(img: &NormalizedImage, path: &Path) -> anyhow::Result<()> {
    // Denormalize: inverse of our normalization formula
    // x_norm ∈ [-1,1] → (x_norm + 1.0) * 127.5 ∈ [0, 255]
    let pixels: Vec<u8> = img.data
        .iter()
        .map(|&val| {
            let denorm = (val + 1.0) * 127.5;
            // Clamp is critical: noise addition can push values to -1.05 or 1.03
            // Without clamping, the u8 cast wraps around (255 → 0), creating
            denorm.clamp(0.0, 255.0) as u8
        })
        .collect();

    // Construct an ImageBuffer from raw pixel data
    let buf: RgbImage = ImageBuffer::from_raw(img.width, img.height, pixels)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer — size mismatch"))?;

    buf.save(path)
        .map_err(|e| anyhow::anyhow!("Failed to save image to {:?}: {}", path, e))?;

    Ok(())
}

// Blend two normalized images using the diffusion equation weights.
// This is a helper used inside the diffusion engine.
//
// result[i] = signal_scale * image[i] + noise_scale * noise[i]
//
// This is exactly: x_t = sqrt(α_t) * x_0 + sqrt(1−α_t) * ε
pub fn blend(image: &[f64], noise: &[f64], signal_scale: f64, noise_scale: f64) -> Vec<f64> {
    assert_eq!(image.len(), noise.len(), "Image and noise must be same size");

    image.iter()
        .zip(noise.iter())
        .map(|(&x0, &epsilon)| {
            // The core diffusion equation, applied pixel-by-pixel
            signal_scale * x0 + noise_scale * epsilon
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blend_at_t0_is_identity() {
        // At t=0: signal_scale=1, noise_scale=0 → output = input
        let image = vec![0.5, -0.3, 0.8];
        let noise = vec![1.0, 1.0, 1.0];
        let result = blend(&image, &noise, 1.0, 0.0);
        for (r, &orig) in result.iter().zip(image.iter()) {
            assert!((r - orig).abs() < 1e-10, "At t=0, output must equal input");
        }
    }

    #[test]
    fn test_blend_at_tmax_is_noise() {
        // At t=T: signal_scale=0, noise_scale=1 → output = noise
        let image = vec![0.5, -0.3, 0.8];
        let noise = vec![0.1, 0.2, 0.3];
        let result = blend(&image, &noise, 0.0, 1.0);
        for (r, &n) in result.iter().zip(noise.iter()) {
            assert!((r - n).abs() < 1e-10, "At t=T, output must equal noise");
        }
    }
}