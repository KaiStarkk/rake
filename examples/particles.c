/**
 * Particle simulation in C (scalar baseline for comparison)
 *
 * This demonstrates the sequential approach that compilers may auto-vectorize.
 * Compare with Rake's explicit SIMD-first design.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define NUM_PARTICLES 1000000
#define NUM_ITERATIONS 100

// Structure of Arrays (SoA) layout - cache-friendly for SIMD
typedef struct {
    float *x, *y, *z;
    float *vx, *vy, *vz;
    float *mass;
} Particles;

// Allocate aligned memory for SIMD
static float* alloc_aligned(size_t count) {
    return (float*)aligned_alloc(32, count * sizeof(float));
}

static Particles create_particles(size_t count) {
    Particles p;
    p.x = alloc_aligned(count);
    p.y = alloc_aligned(count);
    p.z = alloc_aligned(count);
    p.vx = alloc_aligned(count);
    p.vy = alloc_aligned(count);
    p.vz = alloc_aligned(count);
    p.mass = alloc_aligned(count);
    return p;
}

static void free_particles(Particles *p) {
    free(p->x); free(p->y); free(p->z);
    free(p->vx); free(p->vy); free(p->vz);
    free(p->mass);
}

// Initialize particles with random positions and velocities
static void init_particles(Particles *p, size_t count) {
    for (size_t i = 0; i < count; i++) {
        p->x[i] = (float)rand() / RAND_MAX * 100.0f;
        p->y[i] = (float)rand() / RAND_MAX * 100.0f;
        p->z[i] = (float)rand() / RAND_MAX * 100.0f;
        p->vx[i] = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;
        p->vy[i] = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;
        p->vz[i] = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;
        p->mass[i] = (float)rand() / RAND_MAX * 2.0f + 0.5f;
    }
}

// Safe square root - handles negative inputs
static inline float safe_sqrt(float x) {
    return x >= 0.0f ? sqrtf(x) : 0.0f;
}

// 3D vector magnitude
static inline float magnitude(float x, float y, float z) {
    return safe_sqrt(x*x + y*y + z*z);
}

// Clamp value between min and max
static inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// Apply gravity to velocity
static inline float apply_gravity(float vy, float mass) {
    const float gravity = 9.81f;
    const float dt = 0.016f;
    return vy - gravity * mass * dt;
}

// Update position from velocity
static inline float update_position(float pos, float vel) {
    const float dt = 0.016f;
    return pos + vel * dt;
}

// Bounce off boundary - returns new position
static inline float bounce_pos(float pos, float vel, float limit) {
    if (pos < 0.0f) return -pos;
    if (pos > limit) return limit - (pos - limit);
    return pos;
}

// Bounce off boundary - returns new velocity
static inline float bounce_vel(float pos, float vel, float limit) {
    if (pos < 0.0f || pos > limit) return -vel * 0.8f;
    return vel;
}

// Main update loop - this is what compilers will try to auto-vectorize
static void update_particles(Particles *p, size_t count) {
    const float limit = 100.0f;

    // Loop over all particles - compiler may vectorize this
    #pragma omp simd
    for (size_t i = 0; i < count; i++) {
        // Apply gravity to y velocity
        p->vy[i] = apply_gravity(p->vy[i], p->mass[i]);

        // Update positions
        p->x[i] = update_position(p->x[i], p->vx[i]);
        p->y[i] = update_position(p->y[i], p->vy[i]);
        p->z[i] = update_position(p->z[i], p->vz[i]);

        // Handle boundary bouncing for x
        float new_vx = bounce_vel(p->x[i], p->vx[i], limit);
        p->x[i] = bounce_pos(p->x[i], p->vx[i], limit);
        p->vx[i] = new_vx;

        // Handle boundary bouncing for y
        float new_vy = bounce_vel(p->y[i], p->vy[i], limit);
        p->y[i] = bounce_pos(p->y[i], p->vy[i], limit);
        p->vy[i] = new_vy;

        // Handle boundary bouncing for z
        float new_vz = bounce_vel(p->z[i], p->vz[i], limit);
        p->z[i] = bounce_pos(p->z[i], p->vz[i], limit);
        p->vz[i] = new_vz;
    }
}

// Compute total kinetic energy (reduction operation)
static float compute_kinetic_energy(const Particles *p, size_t count) {
    float total = 0.0f;
    #pragma omp simd reduction(+:total)
    for (size_t i = 0; i < count; i++) {
        float speed_sq = p->vx[i]*p->vx[i] + p->vy[i]*p->vy[i] + p->vz[i]*p->vz[i];
        total += 0.5f * p->mass[i] * speed_sq;
    }
    return total;
}

// Distance between two points
static inline float distance(float x1, float y1, float z1,
                            float x2, float y2, float z2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;
    return magnitude(dx, dy, dz);
}

// Dot product
static inline float dot(float x1, float y1, float z1,
                       float x2, float y2, float z2) {
    return x1*x2 + y1*y2 + z1*z2;
}

// Reflect velocity off surface normal
static inline void reflect(float *vx, float *vy, float *vz,
                          float nx, float ny, float nz) {
    float d = dot(*vx, *vy, *vz, nx, ny, nz);
    *vx = *vx - 2.0f * d * nx;
    *vy = *vy - 2.0f * d * ny;
    *vz = *vz - 2.0f * d * nz;
}

int main(int argc, char **argv) {
    size_t num_particles = NUM_PARTICLES;
    int num_iterations = NUM_ITERATIONS;

    if (argc > 1) num_particles = (size_t)atoi(argv[1]);
    if (argc > 2) num_iterations = atoi(argv[2]);

    printf("C Particle Simulation (scalar/auto-vectorized)\n");
    printf("Particles: %zu, Iterations: %d\n", num_particles, num_iterations);

    // Initialize
    srand(42);  // Fixed seed for reproducibility
    Particles p = create_particles(num_particles);
    init_particles(&p, num_particles);

    // Warm-up
    for (int i = 0; i < 10; i++) {
        update_particles(&p, num_particles);
    }

    // Benchmark
    clock_t start = clock();
    for (int iter = 0; iter < num_iterations; iter++) {
        update_particles(&p, num_particles);
    }
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double particles_per_sec = (double)num_particles * num_iterations / elapsed;

    printf("Time: %.3f seconds\n", elapsed);
    printf("Performance: %.2f million particles/sec\n", particles_per_sec / 1e6);

    // Verify result
    float energy = compute_kinetic_energy(&p, num_particles);
    printf("Final kinetic energy: %.2f\n", energy);

    free_particles(&p);
    return 0;
}
