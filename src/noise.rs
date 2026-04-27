//Gaussian (Normal) Distribution:
//
// A random variable ε ~ N(0, 1) has:
//   - Mean 0:     E[ε] = 0
//   - Variance 1: E[ε²] = 1
//   - PDF: f(x) = (1/√(2π)) · exp(−x²/2)
//
// For image diffusion, we need ε with the same shape as the image.
// Each pixel gets its own independent Gaussian sample.

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

// Gaussian noise generator.
// Wraps a random number generator (RNG) and a normal distribution N(0,1).

pub struct GaussianNoise {
    // The N(0,1) distribution — mean=0, std_dev=1
    distribution: Normal<f64>,
    // Seeded RNG for reproducibility
    rng: rand::rngs::StdRng,
}

impl GaussianNoise {
    // Create a new generator.
    /// `seed`: if Some(s), produces reproducible noise. If None, uses OS entropy.
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None    => rand::rngs::StdRng::from_entropy(),
        };
        // Normal::new(mean, std_dev) — we want N(0,1)
        let distribution = Normal::new(0.0, 1.0)
            .expect("Invalid normal distribution parameters");
        Self { distribution, rng }
    }

    /// Sample one scalar from N(0, 1).
    /// This is one "pixel" worth of Gaussian noise (ε_i).
    pub fn sample(&mut self) -> f64 {
        self.distribution.sample(&mut self.rng)
    }

    /// Generate a vector of `n` i.i.d. samples from N(0, 1).
    /// This creates the full epsilon noise tensor for an image.
    ///
    /// "i.i.d." = independently and identically distributed.
    /// Each pixel gets its own sample, independent of every other pixel.
    pub fn sample_vector(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.sample()).collect()
    }

    /// Generate noise with shape matching an image (height × width × channels).
    /// Returns a flat Vec in row-major order: [R00, G00, B00, R01, G01, B01, ...]
    pub fn sample_image_noise(&mut self, width: u32, height: u32, channels: u32) -> Vec<f64> {
        let n = (width * height * channels) as usize;
        self.sample_vector(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_mean_near_zero() {
        let mut gen = GaussianNoise::new(Some(42));
        let samples = gen.sample_vector(100_000);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        // Mean should be close to 0 for large sample sizes
        assert!(mean.abs() < 0.01, "Mean was {mean}, expected ~0");
    }

    #[test]
    fn test_gaussian_std_near_one() {
        let mut gen = GaussianNoise::new(Some(42));
        let samples = gen.sample_vector(100_000);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / samples.len() as f64;
        let std_dev = variance.sqrt();
        assert!((std_dev - 1.0).abs() < 0.01, "Std dev was {std_dev}, expected ~1");
    }

    #[test]
    fn test_reproducible_with_seed() {
        let mut gen1 = GaussianNoise::new(Some(99));
        let mut gen2 = GaussianNoise::new(Some(99));
        let s1 = gen1.sample_vector(100);
        let s2 = gen2.sample_vector(100);
        assert_eq!(s1, s2, "Same seed should produce same samples");
    }
}