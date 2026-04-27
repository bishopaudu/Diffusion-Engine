// THE FORWARD DIFFUSION PROCESS:
//
//   x_t = sqrt(α_t) · x_0 + sqrt(1 − α_t) · ε
//
// We apply this equation at multiple timesteps t ∈ {0, T/steps, 2T/steps, ..., T}
// to generate `num_samples` different noisy variations of the input image.
//
// Each sample:
//   - Uses the SAME α_t (same noise level for a given t)
//   - Uses DIFFERENT ε (different random noise realization)
//   → Gives us diverse augmentations at each noise level

use crate::image_utils::{blend, NormalizedImage};
use crate::noise::GaussianNoise;
use crate::scheduler::NoiseSchedule;

/// Configuration for a diffusion run
pub struct DiffusionConfig {
    /// Total timesteps in the schedule
    pub total_steps: usize,
    /// How many output images to generate
    pub num_samples: usize,
    /// Optional seed for reproducibility
    pub seed: Option<u64>,
}

/// A single diffusion result: a noisy image at a specific timestep
pub struct DiffusionResult {
    /// The noisy image x_t
    pub image: NormalizedImage,
    /// Which timestep this represents
    pub timestep: usize,
    /// The α_t value used (for logging/inspection)
    pub alpha: f64,
    /// Which sample number (0..num_samples)
    pub sample_index: usize,
}

/// Apply the forward diffusion process to a source image.
///
/// # How it works
///
/// 1. Build a noise schedule over `total_steps` timesteps
/// 2. Choose `num_samples` evenly-spaced timesteps to snapshot
/// 3. For each (timestep, sample):
///    a. Look up sqrt(α_t) and sqrt(1−α_t) from the schedule
///    b. Generate fresh Gaussian noise ε with same shape as image
///    c. Compute x_t = sqrt(α_t)·x_0 + sqrt(1−α_t)·ε
///    d. Store the result
pub fn apply_diffusion(
    source: &NormalizedImage,
    config: &DiffusionConfig,
    schedule: &NoiseSchedule,
) -> Vec<DiffusionResult> {
    let mut results = Vec::new();

    // We generate noise fresh for each sample that is different seeds per sample
    // Choose which timesteps to snapshot.
    // If total_steps=1000, num_samples=10, we snapshot at:
    //   t = 0, 111, 222, 333, 444, 555, 666, 777, 888, 999
    // This gives us a visual progression from clean to noisy.
    let timesteps: Vec<usize> = (0..config.num_samples)
        .map(|i| {
            let frac = i as f64 / (config.num_samples - 1).max(1) as f64;
            (frac * (config.total_steps - 1) as f64).round() as usize
        })
        .collect();

    // Image tensor size: total number of scalar values
    let n_pixels = source.len();

    for (sample_idx, &t) in timesteps.iter().enumerate() {
        // Create a noise generator for this sample.
        // We derive the seed from the base seed + sample index so that:
        //   - Each sample gets different noise (different index → different seed)
        //   - Same run produces same results (deterministic if seed provided)
        let sample_seed = config.seed.map(|s| s.wrapping_add(sample_idx as u64));
        let mut noise_gen = GaussianNoise::new(sample_seed);

        // Generate ε ~ N(0, I): one Gaussian sample per pixel per channel
        // This is a random vector in R^(H × W × C)
        let epsilon: Vec<f64> = noise_gen.sample_vector(n_pixels);

        // Look up the precomputed scale factors for this timestep
        let signal_scale = schedule.signal_scale(t); // sqrt(α_t)
        let noise_scale = schedule.noise_scale(t);   // sqrt(1 − α_t)

        // Apply the diffusion equation: x_t = sqrt(α_t)·x_0 + sqrt(1−α_t)·ε
        let noisy_pixels = blend(&source.data, &epsilon, signal_scale, noise_scale);

        let noisy_image = NormalizedImage {
            data: noisy_pixels,
            width: source.width,
            height: source.height,
            channels: source.channels,
        };

        results.push(DiffusionResult {
            image: noisy_image,
            timestep: t,
            alpha: schedule.alphas_cumprod[t],
            sample_index: sample_idx,
        });
    }

    results
}

/// Print a text-based visualization of the noise schedule to the terminal.
/// Shows how α_t (signal retention) drops across timesteps.
pub fn print_schedule_visualization(schedule: &NoiseSchedule, width: usize) {
    println!("\nNoise Schedule: Signal Retention (α_t) over Timesteps");
    println!("{}", "─".repeat(width + 20));

    let steps_to_show = 20; // Show 20 evenly spaced points
    for i in 0..steps_to_show {
        let t = i * schedule.num_timesteps / steps_to_show;
        let alpha = schedule.alphas_cumprod[t];
        let bar_len = (alpha * width as f64).round() as usize;

        // ASCII bar chart showing how much signal remains
        let bar = "█".repeat(bar_len) + &"░".repeat(width - bar_len);
        println!("t={:>4}  α={:.4}  [{}]", t, alpha, bar);
    }
    println!("{}", "─".repeat(width + 20));
    println!("█ = signal retained, ░ = noise territory\n");
}