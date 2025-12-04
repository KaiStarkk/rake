/**
 * Test harness for benchmarking Rake-compiled functions
 *
 * This links against the compiled Rake object file and compares
 * performance with equivalent C implementations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define NUM_PARTICLES 1000000
#define NUM_ITERATIONS 100
#define VECTOR_WIDTH 8

// Rake function declarations (from compiled .o file)
extern __m256 add(__m256 a, __m256 b);
extern __m256 sub(__m256 a, __m256 b);
extern __m256 mul(__m256 a, __m256 b);
extern __m256 safe_sqrt(__m256 x);
extern __m256 magnitude(__m256 x, __m256 y, __m256 z);
extern __m256 clamp(__m256 x, __m256 min, __m256 max);
extern __m256 apply_gravity(__m256 vy, __m256 mass);
extern __m256 update_position(__m256 pos, __m256 vel);
extern __m256 bounce_pos(__m256 pos, __m256 vel, __m256 limit);
extern __m256 bounce_vel(__m256 pos, __m256 vel, __m256 limit);

// SoA particle data (aligned for AVX)
typedef struct {
    float *x, *y, *z;
    float *vx, *vy, *vz;
    float *mass;
} Particles;

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

static void init_particles(Particles *p, size_t count) {
    srand(42);
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

// Update using Rake-compiled functions
static void update_particles_rake(Particles *p, size_t count) {
    __m256 limit = _mm256_set1_ps(100.0f);

    for (size_t i = 0; i < count; i += VECTOR_WIDTH) {
        // Load vectors
        __m256 x = _mm256_load_ps(&p->x[i]);
        __m256 y = _mm256_load_ps(&p->y[i]);
        __m256 z = _mm256_load_ps(&p->z[i]);
        __m256 vx = _mm256_load_ps(&p->vx[i]);
        __m256 vy = _mm256_load_ps(&p->vy[i]);
        __m256 vz = _mm256_load_ps(&p->vz[i]);
        __m256 mass = _mm256_load_ps(&p->mass[i]);

        // Apply gravity (using Rake function)
        vy = apply_gravity(vy, mass);

        // Update positions (using Rake function)
        x = update_position(x, vx);
        y = update_position(y, vy);
        z = update_position(z, vz);

        // Bounce off boundaries (using Rake functions)
        __m256 new_vx = bounce_vel(x, vx, limit);
        x = bounce_pos(x, vx, limit);
        vx = new_vx;

        __m256 new_vy = bounce_vel(y, vy, limit);
        y = bounce_pos(y, vy, limit);
        vy = new_vy;

        __m256 new_vz = bounce_vel(z, vz, limit);
        z = bounce_pos(z, vz, limit);
        vz = new_vz;

        // Store results
        _mm256_store_ps(&p->x[i], x);
        _mm256_store_ps(&p->y[i], y);
        _mm256_store_ps(&p->z[i], z);
        _mm256_store_ps(&p->vx[i], vx);
        _mm256_store_ps(&p->vy[i], vy);
        _mm256_store_ps(&p->vz[i], vz);
    }
}

// C scalar baseline (for comparison)
static inline float c_apply_gravity(float vy, float mass) {
    return vy - 9.81f * mass * 0.016f;
}

static inline float c_update_position(float pos, float vel) {
    return pos + vel * 0.016f;
}

static inline float c_bounce_pos(float pos, float vel, float limit) {
    if (pos < 0.0f) return -pos;
    if (pos > limit) return limit - (pos - limit);
    return pos;
}

static inline float c_bounce_vel(float pos, float vel, float limit) {
    if (pos < 0.0f || pos > limit) return -vel * 0.8f;
    return vel;
}

static void update_particles_c(Particles *p, size_t count) {
    const float limit = 100.0f;

    for (size_t i = 0; i < count; i++) {
        p->vy[i] = c_apply_gravity(p->vy[i], p->mass[i]);

        p->x[i] = c_update_position(p->x[i], p->vx[i]);
        p->y[i] = c_update_position(p->y[i], p->vy[i]);
        p->z[i] = c_update_position(p->z[i], p->vz[i]);

        float new_vx = c_bounce_vel(p->x[i], p->vx[i], limit);
        p->x[i] = c_bounce_pos(p->x[i], p->vx[i], limit);
        p->vx[i] = new_vx;

        float new_vy = c_bounce_vel(p->y[i], p->vy[i], limit);
        p->y[i] = c_bounce_pos(p->y[i], p->vy[i], limit);
        p->vy[i] = new_vy;

        float new_vz = c_bounce_vel(p->z[i], p->vz[i], limit);
        p->z[i] = c_bounce_pos(p->z[i], p->vz[i], limit);
        p->vz[i] = new_vz;
    }
}

static float compute_energy(const Particles *p, size_t count) {
    float total = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float speed_sq = p->vx[i]*p->vx[i] + p->vy[i]*p->vy[i] + p->vz[i]*p->vz[i];
        total += 0.5f * p->mass[i] * speed_sq;
    }
    return total;
}

int main(int argc, char **argv) {
    size_t num_particles = NUM_PARTICLES;
    int num_iterations = NUM_ITERATIONS;

    if (argc > 1) num_particles = (size_t)atoi(argv[1]);
    if (argc > 2) num_iterations = atoi(argv[2]);

    // Ensure count is multiple of vector width
    num_particles = (num_particles / VECTOR_WIDTH) * VECTOR_WIDTH;

    printf("=== Rake vs C Benchmark ===\n");
    printf("Particles: %zu, Iterations: %d\n\n", num_particles, num_iterations);

    // Test Rake version
    {
        Particles p = create_particles(num_particles);
        init_particles(&p, num_particles);

        // Warm-up
        for (int i = 0; i < 10; i++) update_particles_rake(&p, num_particles);

        clock_t start = clock();
        for (int i = 0; i < num_iterations; i++) {
            update_particles_rake(&p, num_particles);
        }
        clock_t end = clock();

        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        double perf = (double)num_particles * num_iterations / elapsed / 1e6;

        printf("Rake (compiled SIMD):\n");
        printf("  Time: %.3f s\n", elapsed);
        printf("  Performance: %.2f M particles/sec\n", perf);
        printf("  Final energy: %.2f\n\n", compute_energy(&p, num_particles));

        free_particles(&p);
    }

    // Test C scalar version
    {
        Particles p = create_particles(num_particles);
        init_particles(&p, num_particles);

        // Warm-up
        for (int i = 0; i < 10; i++) update_particles_c(&p, num_particles);

        clock_t start = clock();
        for (int i = 0; i < num_iterations; i++) {
            update_particles_c(&p, num_particles);
        }
        clock_t end = clock();

        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        double perf = (double)num_particles * num_iterations / elapsed / 1e6;

        printf("C (scalar, may auto-vectorize):\n");
        printf("  Time: %.3f s\n", elapsed);
        printf("  Performance: %.2f M particles/sec\n", perf);
        printf("  Final energy: %.2f\n\n", compute_energy(&p, num_particles));

        free_particles(&p);
    }

    return 0;
}
