//for this project we use linear scheduling
//Linear schedule means noise is added at a constant rate per step.

/// Holds the precomputed schedule for all timesteps.
#[derive(Debug, Clone)]
pub struct NoiseSchedule {
    // β_t values: per-step noise addition rates
    pub betas: Vec<f64>,

    // α_t = ᾱ_t = cumulative product of (1 − β_i)
    // This is the key value used in the diffusion equation
    pub alphas_cumprod: Vec<f64>,

    // sqrt(α_t) — precomputed because we use it every timestep
    pub sqrt_alphas_cumprod: Vec<f64>,

    // sqrt(1 − α_t) — precomputed for same reason
    pub sqrt_one_minus_alphas_cumprod: Vec<f64>,

    // Total number of timesteps T
    pub num_timesteps: usize,
}

impl NoiseSchedule {
    // Create a linear noise schedule.
    // * `num_timesteps` - Total number of diffusion steps T (e.g. 1000)
    // * `beta_start`    - Starting noise level (e.g. 0.0001, very small)
    // * `beta_end`      - Ending noise level   (e.g. 0.02, more aggressive)

    pub fn linear(num_timesteps: usize, beta_start: f64, beta_end: f64) -> Self {
        // Step 1: Generate linearly spaced beta values
        // beta[0] = beta_start, beta[T-1] = beta_end
        // Each intermediate value is: beta_start + i * (beta_end - beta_start) / (T - 1)
        let betas: Vec<f64> = (0..num_timesteps)
            .map(|i| {
                let t = i as f64 / (num_timesteps - 1) as f64; // t ∈ [0, 1]
                beta_start + t * (beta_end - beta_start)
            })
            .collect();

        // Step 2: Compute cumulative product of (1 − β_i)
        // alphas_cumprod[t] = ∏_{i=0}^{t} (1 − β_i)
        //
        // We compute this iteratively:
        //   alphas_cumprod[0] = (1 - beta[0])
        //   alphas_cumprod[1] = alphas_cumprod[0] * (1 - beta[1])
    
        let alphas_cumprod: Vec<f64> = betas
            .iter()
            .scan(1.0_f64, |state, &beta| {
                *state *= 1.0 - beta; // multiply running product by (1 - beta_t)
                Some(*state)          // emit the current product
            })
            .collect();

        // Step 3: Precompute sqrt(α_t) and sqrt(1 − α_t)
        // We do this once here rather than inside the inner loop of diffusion,
        // because taking sqrt() is relatively expensive and we reuse these values
        // across many pixels and many image samples.
        let sqrt_alphas_cumprod: Vec<f64> = alphas_cumprod
            .iter()
            .map(|&a| a.sqrt())
            .collect();

        let sqrt_one_minus_alphas_cumprod: Vec<f64> = alphas_cumprod
            .iter()
            .map(|&a| (1.0 - a).sqrt())
            .collect();

        Self {
            betas,
            alphas_cumprod,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            num_timesteps,
        }
    }

    // Returns the signal scale at timestep t: sqrt(α_t)
    //This is how much of the original image survives
    pub fn signal_scale(&self, t: usize) -> f64 {
        self.sqrt_alphas_cumprod[t]
    }

    //Returns the noise scale at timestep t: sqrt(1 − α_t)
    //This is how much random noise is mixed in
    pub fn noise_scale(&self, t: usize) -> f64 {
        self.sqrt_one_minus_alphas_cumprod[t]
    }

    /// Prints a summary of the schedule for debugging / education
    pub fn describe(&self) {
        println!("Noise Schedule Summary:");
        println!("  Timesteps:  {}", self.num_timesteps);
        println!("  β range:    [{:.6}, {:.6}]", self.betas[0], self.betas[self.num_timesteps - 1]);
        println!("  α_t at t=0: {:.6} (image mostly intact)", self.alphas_cumprod[0]);
        println!("  α_t at t=T: {:.6} (mostly noise)", self.alphas_cumprod[self.num_timesteps - 1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_decreases_monotonically() {
        let schedule = NoiseSchedule::linear(100, 0.0001, 0.02);
        // α_t must strictly decrease: more time = more noise = less signal
        for t in 1..schedule.num_timesteps {
            assert!(
                schedule.alphas_cumprod[t] < schedule.alphas_cumprod[t - 1],
                "Alpha must decrease at every step"
            );
        }
    }

    #[test]
    fn test_alpha_starts_near_one() {
        let schedule = NoiseSchedule::linear(100, 0.0001, 0.02);
        // At t=0, nearly no noise has been added, so α ≈ 1
        assert!(schedule.alphas_cumprod[0] > 0.99);
    }

    #[test]
    fn test_alpha_ends_near_zero() {
        let schedule = NoiseSchedule::linear(1000, 0.0001, 0.02);
        // At t=T, image should be destroyed (α ≈ 0)
        assert!(schedule.alphas_cumprod[999] < 0.02);
    }
}