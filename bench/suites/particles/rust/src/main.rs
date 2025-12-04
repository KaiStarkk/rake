//! Particle simulation in Rust (scalar baseline for comparison)
//!
//! This demonstrates the sequential approach that LLVM may auto-vectorize.
//! Compare with Rake's explicit SIMD-first design.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

const NUM_PARTICLES: usize = 1_000_000;
const NUM_ITERATIONS: usize = 100;

/// Structure of Arrays (SoA) layout - cache-friendly for SIMD
#[derive(Clone)]
struct Particles {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    vx: Vec<f32>,
    vy: Vec<f32>,
    vz: Vec<f32>,
    mass: Vec<f32>,
}

impl Particles {
    fn new(count: usize) -> Self {
        Self {
            x: vec![0.0; count],
            y: vec![0.0; count],
            z: vec![0.0; count],
            vx: vec![0.0; count],
            vy: vec![0.0; count],
            vz: vec![0.0; count],
            mass: vec![0.0; count],
        }
    }

    fn init(&mut self, rng: &mut impl Rng) {
        for i in 0..self.x.len() {
            self.x[i] = rng.gen::<f32>() * 100.0;
            self.y[i] = rng.gen::<f32>() * 100.0;
            self.z[i] = rng.gen::<f32>() * 100.0;
            self.vx[i] = (rng.gen::<f32>() - 0.5) * 10.0;
            self.vy[i] = (rng.gen::<f32>() - 0.5) * 10.0;
            self.vz[i] = (rng.gen::<f32>() - 0.5) * 10.0;
            self.mass[i] = rng.gen::<f32>() * 2.0 + 0.5;
        }
    }
}

/// Safe square root - handles negative inputs
#[inline(always)]
fn safe_sqrt(x: f32) -> f32 {
    if x >= 0.0 { x.sqrt() } else { 0.0 }
}

/// 3D vector magnitude
#[inline(always)]
fn magnitude(x: f32, y: f32, z: f32) -> f32 {
    safe_sqrt(x*x + y*y + z*z)
}

/// Clamp value between min and max
#[inline(always)]
fn clamp(x: f32, min: f32, max: f32) -> f32 {
    if x < min { min }
    else if x > max { max }
    else { x }
}

/// Apply gravity to velocity
#[inline(always)]
fn apply_gravity(vy: f32, mass: f32) -> f32 {
    const GRAVITY: f32 = 9.81;
    const DT: f32 = 0.016;
    vy - GRAVITY * mass * DT
}

/// Update position from velocity
#[inline(always)]
fn update_position(pos: f32, vel: f32) -> f32 {
    const DT: f32 = 0.016;
    pos + vel * DT
}

/// Bounce off boundary - returns new position
#[inline(always)]
fn bounce_pos(pos: f32, _vel: f32, limit: f32) -> f32 {
    if pos < 0.0 { -pos }
    else if pos > limit { limit - (pos - limit) }
    else { pos }
}

/// Bounce off boundary - returns new velocity
#[inline(always)]
fn bounce_vel(pos: f32, vel: f32, limit: f32) -> f32 {
    if pos < 0.0 || pos > limit { -vel * 0.8 }
    else { vel }
}

/// Main update loop - this is what LLVM will try to auto-vectorize
fn update_particles(p: &mut Particles) {
    const LIMIT: f32 = 100.0;
    let count = p.x.len();

    // Loop over all particles - compiler may vectorize this
    for i in 0..count {
        // Apply gravity to y velocity
        p.vy[i] = apply_gravity(p.vy[i], p.mass[i]);

        // Update positions
        p.x[i] = update_position(p.x[i], p.vx[i]);
        p.y[i] = update_position(p.y[i], p.vy[i]);
        p.z[i] = update_position(p.z[i], p.vz[i]);

        // Handle boundary bouncing for x
        let new_vx = bounce_vel(p.x[i], p.vx[i], LIMIT);
        p.x[i] = bounce_pos(p.x[i], p.vx[i], LIMIT);
        p.vx[i] = new_vx;

        // Handle boundary bouncing for y
        let new_vy = bounce_vel(p.y[i], p.vy[i], LIMIT);
        p.y[i] = bounce_pos(p.y[i], p.vy[i], LIMIT);
        p.vy[i] = new_vy;

        // Handle boundary bouncing for z
        let new_vz = bounce_vel(p.z[i], p.vz[i], LIMIT);
        p.z[i] = bounce_pos(p.z[i], p.vz[i], LIMIT);
        p.vz[i] = new_vz;
    }
}

/// Compute total kinetic energy (reduction operation)
fn compute_kinetic_energy(p: &Particles) -> f32 {
    let mut total = 0.0f32;
    for i in 0..p.x.len() {
        let speed_sq = p.vx[i]*p.vx[i] + p.vy[i]*p.vy[i] + p.vz[i]*p.vz[i];
        total += 0.5 * p.mass[i] * speed_sq;
    }
    total
}

/// Distance between two points
#[inline(always)]
fn distance(x1: f32, y1: f32, z1: f32, x2: f32, y2: f32, z2: f32) -> f32 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let dz = z2 - z1;
    magnitude(dx, dy, dz)
}

/// Dot product
#[inline(always)]
fn dot(x1: f32, y1: f32, z1: f32, x2: f32, y2: f32, z2: f32) -> f32 {
    x1*x2 + y1*y2 + z1*z2
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_particles = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(NUM_PARTICLES);
    let num_iterations = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(NUM_ITERATIONS);

    println!("Rust Particle Simulation (scalar/auto-vectorized)");
    println!("Particles: {}, Iterations: {}", num_particles, num_iterations);

    // Initialize with fixed seed for reproducibility
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut particles = Particles::new(num_particles);
    particles.init(&mut rng);

    // Warm-up
    for _ in 0..10 {
        update_particles(&mut particles);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..num_iterations {
        update_particles(&mut particles);
    }
    let elapsed = start.elapsed();

    let elapsed_secs = elapsed.as_secs_f64();
    let particles_per_sec = (num_particles * num_iterations) as f64 / elapsed_secs;

    println!("Time: {:.3} seconds", elapsed_secs);
    println!("Performance: {:.2} million particles/sec", particles_per_sec / 1e6);

    // Verify result
    let energy = compute_kinetic_energy(&particles);
    println!("Final kinetic energy: {:.2}", energy);
}
