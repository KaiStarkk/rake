// Test harness for Rake ray-sphere intersection
// Compares Rake-generated code against scalar C baseline

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

// Rake-generated function (from intersect_flat.rk)
extern __m256 intersect_flat(
    __m256 ray_ox, __m256 ray_oy, __m256 ray_oz,
    __m256 ray_dx, __m256 ray_dy, __m256 ray_dz,
    float sphere_cx, float sphere_cy, float sphere_cz, float sphere_r
);

// Scalar C baseline for comparison
float intersect_scalar(
    float ray_ox, float ray_oy, float ray_oz,
    float ray_dx, float ray_dy, float ray_dz,
    float sphere_cx, float sphere_cy, float sphere_cz, float sphere_r
) {
    // Vector from ray origin to sphere center
    float ocx = ray_ox - sphere_cx;
    float ocy = ray_oy - sphere_cy;
    float ocz = ray_oz - sphere_cz;

    // Quadratic coefficients
    float a = ray_dx * ray_dx + ray_dy * ray_dy + ray_dz * ray_dz;
    float b = 2.0f * (ocx * ray_dx + ocy * ray_dy + ocz * ray_dz);
    float c = ocx * ocx + ocy * ocy + ocz * ocz - sphere_r * sphere_r;

    float disc = b * b - 4.0f * a * c;

    if (disc < 0.0f) {
        return -1.0f;
    }

    float sqrt_disc = sqrtf(disc);
    float t = (-b - sqrt_disc) / (2.0f * a);
    return t;
}

// Auto-vectorized version (let compiler try)
void intersect_auto_vec(
    float* ray_ox, float* ray_oy, float* ray_oz,
    float* ray_dx, float* ray_dy, float* ray_dz,
    float sphere_cx, float sphere_cy, float sphere_cz, float sphere_r,
    float* t_out, int n
) {
    for (int i = 0; i < n; i++) {
        t_out[i] = intersect_scalar(
            ray_ox[i], ray_oy[i], ray_oz[i],
            ray_dx[i], ray_dy[i], ray_dz[i],
            sphere_cx, sphere_cy, sphere_cz, sphere_r
        );
    }
}

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main() {
    const int NUM_RAYS = 8 * 1000000;  // 8 million rays
    const int ITERATIONS = 100;

    // Allocate aligned memory
    float *ray_ox = aligned_alloc(32, NUM_RAYS * sizeof(float));
    float *ray_oy = aligned_alloc(32, NUM_RAYS * sizeof(float));
    float *ray_oz = aligned_alloc(32, NUM_RAYS * sizeof(float));
    float *ray_dx = aligned_alloc(32, NUM_RAYS * sizeof(float));
    float *ray_dy = aligned_alloc(32, NUM_RAYS * sizeof(float));
    float *ray_dz = aligned_alloc(32, NUM_RAYS * sizeof(float));
    float *t_rake = aligned_alloc(32, NUM_RAYS * sizeof(float));
    float *t_c = aligned_alloc(32, NUM_RAYS * sizeof(float));

    // Sphere at origin with radius 1
    float sphere_cx = 0.0f, sphere_cy = 0.0f, sphere_cz = 0.0f;
    float sphere_r = 1.0f;

    // Generate random rays - some hit, some miss (realistic divergence)
    srand(42);
    for (int i = 0; i < NUM_RAYS; i++) {
        // Random origin outside sphere
        float theta = (float)rand() / RAND_MAX * 2.0f * 3.14159f;
        float phi = (float)rand() / RAND_MAX * 3.14159f;
        float r = 5.0f + (float)rand() / RAND_MAX * 5.0f;

        ray_ox[i] = r * sinf(phi) * cosf(theta);
        ray_oy[i] = r * sinf(phi) * sinf(theta);
        ray_oz[i] = r * cosf(phi);

        // Some rays point toward sphere, some miss completely (50/50 divergence)
        if (rand() % 2 == 0) {
            // Point toward origin with jitter
            float jitter = 0.3f;
            ray_dx[i] = -ray_ox[i] + jitter * ((float)rand() / RAND_MAX - 0.5f);
            ray_dy[i] = -ray_oy[i] + jitter * ((float)rand() / RAND_MAX - 0.5f);
            ray_dz[i] = -ray_oz[i] + jitter * ((float)rand() / RAND_MAX - 0.5f);
        } else {
            // Point away from origin (guaranteed miss)
            ray_dx[i] = ray_ox[i] + ((float)rand() / RAND_MAX - 0.5f);
            ray_dy[i] = ray_oy[i] + ((float)rand() / RAND_MAX - 0.5f);
            ray_dz[i] = ray_oz[i] + ((float)rand() / RAND_MAX - 0.5f);
        }

        // Normalize direction
        float len = sqrtf(ray_dx[i]*ray_dx[i] + ray_dy[i]*ray_dy[i] + ray_dz[i]*ray_dz[i]);
        ray_dx[i] /= len;
        ray_dy[i] /= len;
        ray_dz[i] /= len;
    }

    printf("Ray-Sphere Intersection Benchmark\n");
    printf("==================================\n");
    printf("Rays: %d | Iterations: %d\n\n", NUM_RAYS, ITERATIONS);

    // Benchmark Rake version
    double rake_start = get_time_ms();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < NUM_RAYS; i += 8) {
            __m256 ox = _mm256_load_ps(&ray_ox[i]);
            __m256 oy = _mm256_load_ps(&ray_oy[i]);
            __m256 oz = _mm256_load_ps(&ray_oz[i]);
            __m256 dx = _mm256_load_ps(&ray_dx[i]);
            __m256 dy = _mm256_load_ps(&ray_dy[i]);
            __m256 dz = _mm256_load_ps(&ray_dz[i]);

            __m256 t = intersect_flat(ox, oy, oz, dx, dy, dz,
                sphere_cx, sphere_cy, sphere_cz, sphere_r);

            _mm256_store_ps(&t_rake[i], t);
        }
    }
    double rake_time = get_time_ms() - rake_start;

    // Benchmark C auto-vectorized version
    double c_start = get_time_ms();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        intersect_auto_vec(ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
            sphere_cx, sphere_cy, sphere_cz, sphere_r, t_c, NUM_RAYS);
    }
    double c_time = get_time_ms() - c_start;

    // Verify results match
    int mismatches = 0;
    int hits_rake = 0, hits_c = 0;
    for (int i = 0; i < NUM_RAYS; i++) {
        if (t_rake[i] > 0) hits_rake++;
        if (t_c[i] > 0) hits_c++;
        if (fabsf(t_rake[i] - t_c[i]) > 0.001f) {
            if (mismatches < 5) {
                printf("Mismatch at %d: rake=%.4f, c=%.4f\n", i, t_rake[i], t_c[i]);
            }
            mismatches++;
        }
    }

    printf("Results:\n");
    printf("  Hits (Rake): %d (%.1f%%)\n", hits_rake, 100.0f * hits_rake / NUM_RAYS);
    printf("  Hits (C):    %d (%.1f%%)\n", hits_c, 100.0f * hits_c / NUM_RAYS);
    printf("  Mismatches:  %d\n\n", mismatches);

    double rake_rays_per_sec = (double)NUM_RAYS * ITERATIONS / rake_time * 1000.0;
    double c_rays_per_sec = (double)NUM_RAYS * ITERATIONS / c_time * 1000.0;

    printf("Performance:\n");
    printf("  Rake:  %.2f ms (%.2f M rays/sec)\n", rake_time, rake_rays_per_sec / 1e6);
    printf("  C:     %.2f ms (%.2f M rays/sec)\n", c_time, c_rays_per_sec / 1e6);
    printf("  Speedup: %.2fx\n", c_time / rake_time);

    free(ray_ox); free(ray_oy); free(ray_oz);
    free(ray_dx); free(ray_dy); free(ray_dz);
    free(t_rake); free(t_c);

    return 0;
}
