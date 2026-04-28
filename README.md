# 🌊 Diffusion Image Augmentation Engine

> A Rust implementation of the **forward diffusion process** from DDPM (Denoising Diffusion Probabilistic Models) — built from first principles to deeply understand how Stable Diffusion works mathematically.

---

## What This Project Is

This is **not** a generative AI model. It is the mathematical engine that sits underneath one.

Every image generation model like Stable Diffusion or DALL-E is built on top of a **forward diffusion process** — a system that systematically destroys images into pure Gaussian noise over many timesteps. Understanding that process deeply is the foundation for understanding all of modern generative AI.

This project implements that process from scratch in Rust, with:

- A **linear noise schedule** (β₀ → β_T) that controls how quickly noise is added
- A **Gaussian noise generator** built on the Box-Muller transform
- The **closed-form diffusion equation** that jumps to any timestep in one step
- A **Ratatui terminal UI** that shows the process in real time
- A full **CLI interface** for batch image augmentation

The result: feed in any image, get back N noisy variations of it at different noise levels — from "slightly grainy" all the way to "pure static."

---

## The Core Mathematics

The entire engine is built around one equation:

```
x_t = √(ᾱ_t) · x₀  +  √(1 − ᾱ_t) · ε
```

| Symbol | Meaning |
|--------|---------|
| `x₀` | Original image pixels, normalized to [−1, 1] |
| `x_t` | Noisy image at timestep t |
| `ᾱ_t` | Signal retention at step t — how much original survives |
| `ε` | Gaussian noise ~ N(0, I), one sample per pixel channel |
| `√(ᾱ_t)` | Signal scale — near 1.0 early (clear image), near 0.0 late |
| `√(1−ᾱ_t)` | Noise scale — near 0.0 early, near 1.0 late (pure noise) |

Where `ᾱ_t` is the **cumulative product** of the noise schedule:

```
ᾱ_t = (1−β₀) · (1−β₁) · (1−β₂) · ... · (1−β_t)
```

And βₜ values increase linearly from `β_start = 0.0001` to `β_end = 0.02` over T timesteps — the original DDPM schedule.

---

## Project Structure

```
diffusion-engine/
├── Cargo.toml              # Dependencies and project metadata
└── src/
    ├── main.rs             # CLI entry point, Ratatui TUI, orchestration
    ├── lib.rs              # Module declarations
    ├── scheduler.rs        # Noise schedule: β values → ᾱ_t cumulative products
    ├── noise.rs            # Gaussian noise generator (Box-Muller transform)
    ├── image_utils.rs      # Image loading, normalization, blending, saving
    └── diffusion.rs        # Core diffusion engine — applies x_t equation
```

### Module Responsibilities

**`scheduler.rs`** — The noise schedule is the backbone of the system. It precomputes all β values, their cumulative products (ᾱ_t), and the square root scale factors for every timestep. These are computed once at startup and reused across all samples — computing sqrt() inside the inner pixel loop would be ~393× slower.

**`noise.rs`** — Wraps Rust's `rand_distr::Normal` distribution to generate independent N(0,1) samples. Accepts an optional seed for reproducibility. Each sample call internally runs the Box-Muller transform: two uniform random numbers → one Gaussian sample via `z = √(−2·ln(u₁)) · cos(2π·u₂)`.

**`image_utils.rs`** — Handles image I/O and the normalization pipeline. Raw pixel values [0, 255] are normalized to [−1, 1] so they share the same numerical scale as the Gaussian noise (centered at 0). The core `blend()` function is a direct implementation of the diffusion equation — one multiply-add per pixel per channel.

**`diffusion.rs`** — The orchestrator. Selects evenly-spaced timesteps across T steps, generates fresh Gaussian noise per sample, looks up precomputed scale factors from the schedule, and calls `blend()` to produce each output image. Each sample gets a derived seed (`base_seed + sample_index`) so they have different noise patterns while remaining reproducible.

**`main.rs`** — Parses CLI arguments with `clap`, validates all inputs before touching any files, builds the schedule, loads the image, runs the engine inside a Ratatui full-screen TUI with a live progress bar and log, then saves all outputs.

---

## Installation

### Prerequisites

- **Rust** 1.70 or later — install from [rustup.rs](https://rustup.rs)
- Any common image format on hand (JPEG, PNG, BMP, TIFF, WebP)

### Build

```bash
git clone https://github.com/yourname/diffusion-engine
cd diffusion-engine
cargo build --release
```

The binary will be at `./target/release/diffusion`.

---

## Usage

### Basic — augment an image with defaults

```bash
diffusion augment photo.jpg
```

Generates 10 output images in `./output/` using T=1000 timesteps.

### Full options

```bash
diffusion augment input.jpg \
  --steps 1000 \
  --samples 10 \
  --out-dir ./augmented \
  --beta-start 0.0001 \
  --beta-end 0.02 \
  --seed 42 \
  --show-schedule true
```

### Options reference

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | `1000` | Total timesteps T in the noise schedule |
| `--samples` | `10` | Number of output images to generate |
| `--out-dir` | `output` | Directory to save output images (created if missing) |
| `--beta-start` | `0.0001` | Starting β value — controls noise at step 0 |
| `--beta-end` | `0.02` | Ending β value — controls noise at step T |
| `--seed` | *(random)* | Integer seed for reproducibility. Omit for different output each run |
| `--show-schedule` | `true` | Print ASCII chart of ᾱ_t decay before running |

### Output filenames

Each output file encodes the timestep and alpha value it was generated at:

```
output/
├── diffused_sample00_t0000_alpha0.9999.png   ← almost no noise
├── diffused_sample01_t0111_alpha0.8421.png
├── diffused_sample02_t0222_alpha0.6203.png
├── diffused_sample03_t0333_alpha0.4018.png
├── diffused_sample04_t0444_alpha0.2251.png
├── diffused_sample05_t0555_alpha0.1089.png
├── diffused_sample06_t0666_alpha0.0443.png
├── diffused_sample07_t0777_alpha0.0148.png
├── diffused_sample08_t0888_alpha0.0038.png
└── diffused_sample09_t0999_alpha0.0006.png   ← almost pure noise
```

---

## Run Tests

```bash
# Run all unit tests
cargo test

# Run tests with output visible
cargo test -- --nocapture
```

Tests cover:

- `scheduler.rs` — α monotonically decreases, starts near 1, ends near 0
- `noise.rs` — empirical mean ≈ 0, std dev ≈ 1 over large samples, seed reproducibility
- `image_utils.rs` — blend at t=0 is identity, blend at t=T is pure noise
- `main.rs` — input validation rejects bad file paths, invalid β ranges, samples > steps

---

## Key Concepts

### Why Gaussian noise?

Gaussian (Normal) noise is the physically correct model for real-world measurement noise — electrical noise in sensors, quantization errors in image capture, atmospheric scattering — all of these are sums of many small independent random processes. By the Central Limit Theorem, such sums always converge to a Gaussian distribution. So when we add Gaussian noise to images, we are simulating the most physically realistic degradation that exists.

### Why the closed-form equation?

The diffusion process was originally defined as a chain of T small Markov steps, each adding a tiny amount of noise. To get `x_t`, you'd need to compute all t steps in sequence. The closed-form equation collapses all T steps into one:

```
x_t = √(ᾱ_t) · x₀ + √(1−ᾱ_t) · ε
```

This works because Gaussian distributions have a special property: the sum of Gaussians is still Gaussian. So the cumulative effect of T Gaussian noise additions can be expressed as a single scaled Gaussian. This makes training diffusion models practical — you don't run 500 forward steps to get a t=500 training sample.

### Why normalize pixels to [−1, 1]?

Our Gaussian noise has mean 0 and standard deviation 1. Normalizing pixels to [−1, 1] centers them at 0 — the same center as the noise. This symmetry makes the diffusion equation mathematically balanced. If pixels were in [0, 255], the noise scale and signal scale would be in completely different units, and the equation would produce distorted results.

### What does the seed control?

The seed initializes a pseudo-random number generator (PRNG) — a deterministic algorithm that produces a reproducible sequence of numbers that appear random. With the same seed, every pixel of every output image is identical across runs. Different samples get seeds `base_seed + 0`, `base_seed + 1`, `base_seed + 2`, etc., so they each get different noise patterns while the full run remains reproducible.

---

## Understanding the Output

Each output image is a blend of the original and pure Gaussian noise, weighted by the timestep:

| α_t value | What the image looks like |
|-----------|--------------------------|
| > 0.9 | Almost identical to original — very faint grain |
| 0.5 – 0.9 | Clearly recognizable but noticeably noisy |
| 0.1 – 0.5 | Heavily degraded — shapes barely visible |
| < 0.1 | Mostly static — only traces of original remain |
| ≈ 0.0 | Pure Gaussian noise — original completely gone |

The `√(α_t)` column in the terminal output tells you exactly what percentage of the original signal survived.

---

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `image` | 0.25 | Image loading (JPEG/PNG/BMP/TIFF/WebP) and saving |
| `rand` | 0.8 | Seedable pseudo-random number generator (StdRng) |
| `rand_distr` | 0.4 | Normal distribution sampler (Box-Muller internally) |
| `ratatui` | 0.26 | Terminal UI framework — progress bar and live log |
| `crossterm` | 0.27 | Cross-platform raw terminal control |
| `clap` | 4 | CLI argument parsing with derive macros |
| `indicatif` | 0.17 | Progress bar helpers |
| `anyhow` | 1 | Flexible error handling and propagation |

---

## Connection to Real Diffusion Models

This project implements the **forward process** `q(x_t | x₀)` from the DDPM paper (Ho et al., 2020). A complete generative model adds:

1. **An encoder** (VAE) that compresses images into a latent space before diffusion
2. **A reverse process** where a U-Net neural network learns to predict the noise ε given x_t
3. **A training loop** using the ELBO loss: `||ε − ε_θ(x_t, t)||²`
4. **A sampler** (DDPM, DDIM, etc.) that iteratively removes noise from pure Gaussian noise to produce new images

Everything in this project is what those models compute underneath. The neural network is just a function approximator trained on top of this exact mathematical process.

---

## Learning Resources

The interactive HTML textbooks built alongside this project explain every concept from first principles:

- **`stats_visual_textbook.html`** — 10-chapter interactive statistics guide covering probability, the Gaussian distribution, sampling, the diffusion equation, and the beta schedule
- **`deep_concepts_explainer.html`** — Deep dives into Gaussian distribution (with the Central Limit Theorem demo), Box-Muller transform geometry and arithmetic, cumulative product decay, and seed mechanics

---

## References

- Ho, J., Jain, A., & Abbeel, P. (2020). **Denoising Diffusion Probabilistic Models**. NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Box, G. E. P., & Muller, M. E. (1958). **A Note on the Generation of Random Normal Deviates**. The Annals of Mathematical Statistics.
- Weng, L. (2021). **What are Diffusion Models?** — Lilian Weng's Blog. [lilianweng.github.io](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

---

## License

MIT License — see `LICENSE` for details.

---

*Built as a learning project to understand the mathematics of diffusion models from first principles. Every line of code in this project maps directly to an equation in the DDPM paper.*
