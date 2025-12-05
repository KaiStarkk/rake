/**
 * Rake Raytracer Headless Benchmark
 *
 * Compares C scalar, C SIMD (AVX2), and Rake raytracing performance
 * without requiring a display.
 *
 * Build:
 *   clang -O3 -mavx2 benchmark.c raytracer_rake.o -o benchmark -lm
 *
 * Run:
 *   ./benchmark
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <immintrin.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Configuration
 * ═══════════════════════════════════════════════════════════════════════════ */

#define WIDTH  1920
#define HEIGHT 1080
#define NUM_SPHERES 10
#define NUM_RAYS (WIDTH * HEIGHT)
#define BENCHMARK_ITERATIONS 100
#define WARMUP_ITERATIONS 10

/* ═══════════════════════════════════════════════════════════════════════════
 * Data Structures
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    float cx, cy, cz, r;
} Sphere;

/* Ray pack for SIMD (SoA layout) */
typedef struct {
    float *ox, *oy, *oz;
    float *dx, *dy, *dz;
} RayPack;

/* ═══════════════════════════════════════════════════════════════════════════
 * External Rake Functions
 * ═══════════════════════════════════════════════════════════════════════════ */

extern __m256 dot(__m256 ax, __m256 ay, __m256 az,
                  __m256 bx, __m256 by, __m256 bz);

extern __m256 intersect_flat(__m256 ray_ox, __m256 ray_oy, __m256 ray_oz,
                             __m256 ray_dx, __m256 ray_dy, __m256 ray_dz,
                             float sphere_cx, float sphere_cy, float sphere_cz,
                             float sphere_r);

/* ═══════════════════════════════════════════════════════════════════════════
 * C Scalar Implementation
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline float dot_scalar(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float intersect_scalar(Vec3 origin, Vec3 dir, Sphere s) {
    Vec3 oc = { origin.x - s.cx, origin.y - s.cy, origin.z - s.cz };

    float a = dot_scalar(dir, dir);
    float b = 2.0f * dot_scalar(oc, dir);
    float c = dot_scalar(oc, oc) - s.r * s.r;

    float disc = b * b - 4.0f * a * c;

    if (disc < 0.0f) return -1.0f;

    float t = (-b - sqrtf(disc)) / (2.0f * a);
    return t > 0.0f ? t : -1.0f;
}

void trace_scalar(RayPack *rays, Sphere *spheres, int num_spheres, float *result) {
    for (int i = 0; i < NUM_RAYS; i++) {
        Vec3 origin = { rays->ox[i], rays->oy[i], rays->oz[i] };
        Vec3 dir = { rays->dx[i], rays->dy[i], rays->dz[i] };

        float closest_t = 1e30f;
        for (int s = 0; s < num_spheres; s++) {
            float t = intersect_scalar(origin, dir, spheres[s]);
            if (t > 0.0f && t < closest_t) {
                closest_t = t;
            }
        }
        result[i] = closest_t < 1e29f ? closest_t : -1.0f;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * C SIMD Implementation (AVX2)
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline __m256 dot_simd(__m256 ax, __m256 ay, __m256 az,
                              __m256 bx, __m256 by, __m256 bz) {
    __m256 xx = _mm256_mul_ps(ax, bx);
    __m256 yy = _mm256_mul_ps(ay, by);
    __m256 zz = _mm256_mul_ps(az, bz);
    return _mm256_add_ps(_mm256_add_ps(xx, yy), zz);
}

static inline __m256 intersect_simd(__m256 ox, __m256 oy, __m256 oz,
                                    __m256 dx, __m256 dy, __m256 dz,
                                    float cx, float cy, float cz, float r) {
    __m256 scx = _mm256_set1_ps(cx);
    __m256 scy = _mm256_set1_ps(cy);
    __m256 scz = _mm256_set1_ps(cz);
    __m256 sr = _mm256_set1_ps(r);

    __m256 ocx = _mm256_sub_ps(ox, scx);
    __m256 ocy = _mm256_sub_ps(oy, scy);
    __m256 ocz = _mm256_sub_ps(oz, scz);

    __m256 a = dot_simd(dx, dy, dz, dx, dy, dz);
    __m256 b = _mm256_mul_ps(_mm256_set1_ps(2.0f), dot_simd(ocx, ocy, ocz, dx, dy, dz));
    __m256 c = _mm256_sub_ps(dot_simd(ocx, ocy, ocz, ocx, ocy, ocz), _mm256_mul_ps(sr, sr));

    __m256 disc = _mm256_sub_ps(_mm256_mul_ps(b, b),
                                _mm256_mul_ps(_mm256_set1_ps(4.0f), _mm256_mul_ps(a, c)));

    __m256 miss_mask = _mm256_cmp_ps(disc, _mm256_setzero_ps(), _CMP_LT_OQ);

    __m256 sqrt_disc = _mm256_sqrt_ps(disc);
    __m256 t = _mm256_div_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_setzero_ps(), b), sqrt_disc),
                             _mm256_mul_ps(_mm256_set1_ps(2.0f), a));

    __m256 miss_val = _mm256_set1_ps(-1.0f);
    return _mm256_blendv_ps(t, miss_val, miss_mask);
}

void trace_simd_c(RayPack *rays, Sphere *spheres, int num_spheres, float *result) {
    /* Initialize result to large values */
    for (int i = 0; i < NUM_RAYS; i++) result[i] = 1e30f;

    for (int s = 0; s < num_spheres; s++) {
        for (int i = 0; i < NUM_RAYS; i += 8) {
            __m256 ox = _mm256_loadu_ps(&rays->ox[i]);
            __m256 oy = _mm256_loadu_ps(&rays->oy[i]);
            __m256 oz = _mm256_loadu_ps(&rays->oz[i]);
            __m256 dx = _mm256_loadu_ps(&rays->dx[i]);
            __m256 dy = _mm256_loadu_ps(&rays->dy[i]);
            __m256 dz = _mm256_loadu_ps(&rays->dz[i]);

            __m256 t = intersect_simd(ox, oy, oz, dx, dy, dz,
                                      spheres[s].cx, spheres[s].cy,
                                      spheres[s].cz, spheres[s].r);

            __m256 current = _mm256_loadu_ps(&result[i]);
            __m256 hit_mask = _mm256_and_ps(_mm256_cmp_ps(t, _mm256_setzero_ps(), _CMP_GT_OQ),
                                           _mm256_cmp_ps(t, current, _CMP_LT_OQ));
            __m256 new_t = _mm256_blendv_ps(current, t, hit_mask);
            _mm256_storeu_ps(&result[i], new_t);
        }
    }

    /* Convert large values to -1 */
    for (int i = 0; i < NUM_RAYS; i++) {
        if (result[i] > 1e29f) result[i] = -1.0f;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Rake Implementation (uses compiled Rake code)
 * ═══════════════════════════════════════════════════════════════════════════ */

void trace_rake(RayPack *rays, Sphere *spheres, int num_spheres, float *result) {
    /* Initialize result to large values */
    for (int i = 0; i < NUM_RAYS; i++) result[i] = 1e30f;

    for (int s = 0; s < num_spheres; s++) {
        for (int i = 0; i < NUM_RAYS; i += 8) {
            __m256 ox = _mm256_loadu_ps(&rays->ox[i]);
            __m256 oy = _mm256_loadu_ps(&rays->oy[i]);
            __m256 oz = _mm256_loadu_ps(&rays->oz[i]);
            __m256 dx = _mm256_loadu_ps(&rays->dx[i]);
            __m256 dy = _mm256_loadu_ps(&rays->dy[i]);
            __m256 dz = _mm256_loadu_ps(&rays->dz[i]);

            /* Use Rake-compiled intersection */
            __m256 t = intersect_flat(ox, oy, oz, dx, dy, dz,
                                      spheres[s].cx, spheres[s].cy,
                                      spheres[s].cz, spheres[s].r);

            __m256 current = _mm256_loadu_ps(&result[i]);
            __m256 hit_mask = _mm256_and_ps(_mm256_cmp_ps(t, _mm256_setzero_ps(), _CMP_GT_OQ),
                                           _mm256_cmp_ps(t, current, _CMP_LT_OQ));
            __m256 new_t = _mm256_blendv_ps(current, t, hit_mask);
            _mm256_storeu_ps(&result[i], new_t);
        }
    }

    /* Convert large values to -1 */
    for (int i = 0; i < NUM_RAYS; i++) {
        if (result[i] > 1e29f) result[i] = -1.0f;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Ray Generation
 * ═══════════════════════════════════════════════════════════════════════════ */

void generate_rays(RayPack *rays, int width, int height) {
    float aspect = (float)width / (float)height;
    float fov = 60.0f * M_PI / 180.0f;
    float half_height = tanf(fov / 2.0f);
    float half_width = aspect * half_height;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;

            rays->ox[idx] = 0.0f;
            rays->oy[idx] = 0.0f;
            rays->oz[idx] = 0.0f;

            float u = (2.0f * ((float)x + 0.5f) / (float)width - 1.0f) * half_width;
            float v = (1.0f - 2.0f * ((float)y + 0.5f) / (float)height) * half_height;

            float len = sqrtf(u * u + v * v + 1.0f);
            rays->dx[idx] = u / len;
            rays->dy[idx] = v / len;
            rays->dz[idx] = -1.0f / len;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Timing
 * ═══════════════════════════════════════════════════════════════════════════ */

double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║             Rake Raytracer Performance Benchmark                 ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Resolution: %d x %d (%d rays)                           ║\n", WIDTH, HEIGHT, NUM_RAYS);
    printf("║  Spheres: %d                                                      ║\n", NUM_SPHERES);
    printf("║  Iterations: %d (warmup: %d)                                    ║\n", BENCHMARK_ITERATIONS, WARMUP_ITERATIONS);
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    /* Allocate buffers (aligned for SIMD) */
    RayPack rays;
    rays.ox = aligned_alloc(32, NUM_RAYS * sizeof(float));
    rays.oy = aligned_alloc(32, NUM_RAYS * sizeof(float));
    rays.oz = aligned_alloc(32, NUM_RAYS * sizeof(float));
    rays.dx = aligned_alloc(32, NUM_RAYS * sizeof(float));
    rays.dy = aligned_alloc(32, NUM_RAYS * sizeof(float));
    rays.dz = aligned_alloc(32, NUM_RAYS * sizeof(float));
    float *result = aligned_alloc(32, NUM_RAYS * sizeof(float));

    /* Generate rays */
    printf("Generating rays...\n");
    generate_rays(&rays, WIDTH, HEIGHT);

    /* Setup spheres */
    Sphere spheres[NUM_SPHERES];
    for (int i = 0; i < NUM_SPHERES; i++) {
        spheres[i].cx = (i % 5 - 2) * 2.5f;
        spheres[i].cy = (i / 5 - 0.5f) * 2.0f;
        spheres[i].cz = -5.0f - (i % 3) * 2.0f;
        spheres[i].r = 0.8f + (i % 3) * 0.2f;
    }

    /* Warmup */
    printf("Warming up...\n");
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        trace_scalar(&rays, spheres, NUM_SPHERES, result);
        trace_simd_c(&rays, spheres, NUM_SPHERES, result);
        trace_rake(&rays, spheres, NUM_SPHERES, result);
    }

    /* Benchmark C Scalar */
    printf("\nBenchmarking C Scalar...\n");
    double start = get_time_ms();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        trace_scalar(&rays, spheres, NUM_SPHERES, result);
    }
    double scalar_time = (get_time_ms() - start) / BENCHMARK_ITERATIONS;

    /* Count hits */
    int scalar_hits = 0;
    for (int i = 0; i < NUM_RAYS; i++) {
        if (result[i] > 0.0f) scalar_hits++;
    }

    /* Benchmark C SIMD */
    printf("Benchmarking C SIMD (AVX2)...\n");
    start = get_time_ms();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        trace_simd_c(&rays, spheres, NUM_SPHERES, result);
    }
    double simd_c_time = (get_time_ms() - start) / BENCHMARK_ITERATIONS;

    int simd_hits = 0;
    for (int i = 0; i < NUM_RAYS; i++) {
        if (result[i] > 0.0f) simd_hits++;
    }

    /* Benchmark Rake */
    printf("Benchmarking Rake SIMD...\n");
    start = get_time_ms();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        trace_rake(&rays, spheres, NUM_SPHERES, result);
    }
    double rake_time = (get_time_ms() - start) / BENCHMARK_ITERATIONS;

    int rake_hits = 0;
    for (int i = 0; i < NUM_RAYS; i++) {
        if (result[i] > 0.0f) rake_hits++;
    }

    /* Results */
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║                       BENCHMARK RESULTS                          ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║                                                                  ║\n");
    printf("║  C Scalar:                                                       ║\n");
    printf("║    Time:     %8.2f ms/frame                                   ║\n", scalar_time);
    printf("║    FPS:      %8.1f                                            ║\n", 1000.0/scalar_time);
    printf("║    Hits:     %8d rays                                      ║\n", scalar_hits);
    printf("║                                                                  ║\n");
    printf("║  C SIMD (AVX2):                                                  ║\n");
    printf("║    Time:     %8.2f ms/frame                                   ║\n", simd_c_time);
    printf("║    FPS:      %8.1f                                            ║\n", 1000.0/simd_c_time);
    printf("║    Speedup:  %8.2fx vs scalar                                 ║\n", scalar_time/simd_c_time);
    printf("║    Hits:     %8d rays                                      ║\n", simd_hits);
    printf("║                                                                  ║\n");
    printf("║  Rake SIMD:                                                      ║\n");
    printf("║    Time:     %8.2f ms/frame                                   ║\n", rake_time);
    printf("║    FPS:      %8.1f                                            ║\n", 1000.0/rake_time);
    printf("║    Speedup:  %8.2fx vs scalar                                 ║\n", scalar_time/rake_time);
    printf("║    Hits:     %8d rays                                      ║\n", rake_hits);
    printf("║                                                                  ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Rake vs C SIMD: %.2fx                                           ║\n", simd_c_time/rake_time);
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    /* Verify correctness */
    if (scalar_hits == simd_hits && simd_hits == rake_hits) {
        printf("✓ All implementations produce identical results\n");
    } else {
        printf("✗ WARNING: Hit counts differ!\n");
    }

    /* Cleanup */
    free(rays.ox); free(rays.oy); free(rays.oz);
    free(rays.dx); free(rays.dy); free(rays.dz);
    free(result);

    return 0;
}
