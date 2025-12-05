/* Test harness for Rake raytracer */
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <string.h>

/* External functions from compiled Rake code */
extern __m256 dot(__m256 ax, __m256 ay, __m256 az,
                  __m256 bx, __m256 by, __m256 bz);

extern __m256 intersect_flat(__m256 ray_ox, __m256 ray_oy, __m256 ray_oz,
                             __m256 ray_dx, __m256 ray_dy, __m256 ray_dz,
                             float sphere_cx, float sphere_cy, float sphere_cz,
                             float sphere_r);

/* Helper to print a vector */
void print_vec(__m256 v, const char* name) {
    float vals[8] __attribute__((aligned(32)));
    _mm256_store_ps(vals, v);
    printf("%s: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
           name, vals[0], vals[1], vals[2], vals[3],
           vals[4], vals[5], vals[6], vals[7]);
}

int main() {
    printf("=== Rake Raytracer Test ===\n\n");

    /* Sphere at origin with radius 1 */
    float cx = 0.0f, cy = 0.0f, cz = 5.0f;
    float r = 1.0f;

    /* 8 rays: some will hit, some will miss */
    /* Ray origins at z=0, looking in +z direction */
    __m256 ray_ox = _mm256_set_ps(-2.0f, -1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f);
    __m256 ray_oy = _mm256_set_ps( 0.0f,  0.0f,  0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    __m256 ray_oz = _mm256_set_ps( 0.0f,  0.0f,  0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    /* All rays pointing in +z direction */
    __m256 ray_dx = _mm256_set_ps( 0.0f,  0.0f,  0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    __m256 ray_dy = _mm256_set_ps( 0.0f,  0.0f,  0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    __m256 ray_dz = _mm256_set_ps( 1.0f,  1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

    printf("Sphere: center=(%.1f, %.1f, %.1f), radius=%.1f\n\n", cx, cy, cz, r);
    printf("Ray origins X: ");
    print_vec(ray_ox, "ox");
    printf("\n");

    /* Test dot product */
    printf("Testing dot product...\n");
    __m256 dot_result = dot(ray_dx, ray_dy, ray_dz, ray_dx, ray_dy, ray_dz);
    print_vec(dot_result, "dot(dir,dir)");  /* Should all be 1.0 */
    printf("\n");

    /* Test intersection */
    printf("Testing ray-sphere intersection...\n");
    __m256 t_result = intersect_flat(ray_ox, ray_oy, ray_oz,
                                     ray_dx, ray_dy, ray_dz,
                                     cx, cy, cz, r);

    print_vec(t_result, "t");
    printf("\nExpected: rays at x=[-1,1] should hit (t>0), others miss (t=-1)\n");

    /* Verify results */
    float t_vals[8] __attribute__((aligned(32)));
    _mm256_store_ps(t_vals, t_result);

    int hits = 0, misses = 0;
    for (int i = 0; i < 8; i++) {
        if (t_vals[i] > 0.0f) {
            hits++;
            printf("Lane %d: HIT at t=%.4f\n", i, t_vals[i]);
        } else {
            misses++;
            printf("Lane %d: MISS\n", i);
        }
    }
    printf("\nTotal: %d hits, %d misses\n", hits, misses);

    return 0;
}
