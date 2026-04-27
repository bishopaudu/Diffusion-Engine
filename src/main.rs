// Entry point for the Diffusion Engine.
// Provides a terminal UI with:
//   1. A Ratatui progress display showing diffusion steps
//   2. Clap CLI for argument parsing
//   3. Orchestration of all modules

use clap::Parser;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph},
    Terminal,
};
use std::{
    io,
    path::{Path, PathBuf},
    time::Duration,
};

// Import our library modules
use diffusion_engine::{
    diffusion::{apply_diffusion, print_schedule_visualization, DiffusionConfig},
    image_utils::{load_and_normalize, save_normalized},
    scheduler::NoiseSchedule,
};

/// CLI argument structure — clap derives the parser automatically
#[derive(Parser, Debug)]
#[command(
    name = "diffusion",
    about = "Diffusion Image Augmentation Engine\nApplies forward diffusion process to generate noisy variations",
    version
)]
struct Args {
    /// Subcommand
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Apply diffusion to an image and save noisy variations
    Augment {
        /// Path to input image (JPEG, PNG, etc.)
        input: PathBuf,

        /// Number of diffusion timesteps in the schedule
        #[arg(long, default_value = "1000")]
        steps: usize,

        /// Number of output image samples to generate
        #[arg(long, default_value = "10")]
        samples: usize,

        /// Output directory for augmented images
        #[arg(long, default_value = "output")]
        out_dir: PathBuf,

        /// Starting beta value for noise schedule
        #[arg(long, default_value = "0.0001")]
        beta_start: f64,

        /// Ending beta value for noise schedule
        #[arg(long, default_value = "0.02")]
        beta_end: f64,

        /// Random seed for reproducibility (omit for random)
        #[arg(long)]
        seed: Option<u64>,

        /// Show schedule visualization before running
        #[arg(long, default_value = "true")]
        show_schedule: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Augment {
            input,
            steps,
            samples,
            out_dir,
            beta_start,
            beta_end,
            seed,
            show_schedule,
        } => {
            run_augmentation(
                &input, steps, samples, &out_dir,
                beta_start, beta_end, seed, show_schedule,
            )?;
        }
    }

    Ok(())
}

fn run_augmentation(
    input_path: &Path,
    steps: usize,
    num_samples: usize,
    out_dir: &Path,
    beta_start: f64,
    beta_end: f64,
    seed: Option<u64>,
    show_schedule: bool,
) -> anyhow::Result<()> {
    // ── Step 1: Setup ────────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════╗");
    println!("║  Diffusion Image Augmentation Engine ║");
    println!("╚══════════════════════════════════════╝\n");

    println!("Loading image: {}", input_path.display());
    let source_image = load_and_normalize(input_path)?;
    println!(
        "  Image loaded: {}×{} pixels ({} values total)",
        source_image.width, source_image.height, source_image.len()
    );

    // ── Step 2: Build noise schedule ─────────────────────────────────────────
    println!("\nBuilding noise schedule ({steps} steps)...");
    let schedule = NoiseSchedule::linear(steps, beta_start, beta_end);
    schedule.describe();

    if show_schedule {
        print_schedule_visualization(&schedule, 40);
    }

    // ── Step 3: Create output directory ──────────────────────────────────────
    std::fs::create_dir_all(out_dir)?;

    // ── Step 4: Run Ratatui UI for diffusion progress ─────────────────────
    let results = run_with_tui(
        &source_image, num_samples, &schedule, seed, steps
    )?;

    // ── Step 5: Save all output images ───────────────────────────────────────
    println!("\nSaving {} augmented images to {}/", results.len(), out_dir.display());

    for result in &results {
        let filename = format!(
            "diffused_sample{:02}_t{:04}_alpha{:.4}.png",
            result.sample_index, result.timestep, result.alpha
        );
        let out_path = out_dir.join(&filename);
        save_normalized(&result.image, &out_path)?;
        println!(
            "  Saved: {} (t={}, α={:.4}, signal={:.1}%, noise={:.1}%)",
            filename,
            result.timestep,
            result.alpha,
            result.alpha.sqrt() * 100.0,
            (1.0 - result.alpha).sqrt() * 100.0
        );
    }

    println!("\nDone! {} images saved.", results.len());
    Ok(())
}

/// Run the diffusion engine inside a Ratatui terminal UI.
/// Shows a progress bar and log of completed samples.
fn run_with_tui(
    source: &diffusion_engine::image_utils::NormalizedImage,
    num_samples: usize,
    schedule: &NoiseSchedule,
    seed: Option<u64>,
    total_steps: usize,
) -> anyhow::Result<Vec<diffusion_engine::diffusion::DiffusionResult>> {
    // Set up terminal for Ratatui
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let config = DiffusionConfig {
        total_steps,
        num_samples,
        seed,
    };

    // Run diffusion (this is fast — no model inference, just math)
    let results = apply_diffusion(source, &config, schedule);

    // Display progress in TUI
    let mut log_items: Vec<ListItem> = Vec::new();
    for (i, result) in results.iter().enumerate() {
        let progress = (i + 1) as f64 / results.len() as f64;
        let log_entry = format!(
            "[{:>2}/{}] t={:>4}  α={:.4}  signal={:.1}%  noise={:.1}%",
            i + 1,
            results.len(),
            result.timestep,
            result.alpha,
            result.alpha.sqrt() * 100.0,
            (1.0 - result.alpha).sqrt() * 100.0,
        );
        log_items.push(ListItem::new(log_entry));

        // Render TUI frame
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),   // title
                    Constraint::Length(3),   // progress bar
                    Constraint::Min(0),      // log
                    Constraint::Length(2),   // footer
                ])
                .split(f.size());

            // Title
            let title = Paragraph::new(format!(
                "  Diffusion Augmentation Engine  |  {}×{}px  |  {} timesteps",
                source.width, source.height, total_steps
            ))
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .block(Block::default().borders(Borders::ALL).title("Status"));
            f.render_widget(title, chunks[0]);

            // Progress bar
            let gauge = Gauge::default()
                .block(Block::default().title("Progress").borders(Borders::ALL))
                .gauge_style(Style::default().fg(Color::Green).bg(Color::DarkGray))
                .ratio(progress)
                .label(format!("{}/{} samples", i + 1, results.len()));
            f.render_widget(gauge, chunks[1]);

            // Log list
            let list = List::new(log_items.clone())
                .block(Block::default().title("Diffusion Log").borders(Borders::ALL))
                .style(Style::default().fg(Color::White));
            f.render_widget(list, chunks[2]);

            // Footer
            let footer = Paragraph::new("  Press 'q' to exit after completion")
                .style(Style::default().fg(Color::DarkGray));
            f.render_widget(footer, chunks[3]);
        })?;

        // Small delay to make progress visible
        std::thread::sleep(Duration::from_millis(80));
    }

    // Wait for 'q' key press
    loop {
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') || key.code == KeyCode::Enter {
                    break;
                }
            }
        }
    }

    // Restore terminal to normal state
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(results)
}