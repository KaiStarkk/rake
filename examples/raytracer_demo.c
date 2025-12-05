/**
 * Rake Raytracer Demo
 *
 * Compares C scalar, C SIMD (AVX2), and Rake raytracing performance.
 * Renders multiple spheres with real-time visualization using SDL2.
 *
 * Build:
 *   ./scripts/build_demo.sh
 *
 * Run:
 *   ./raytracer_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <SDL2/SDL.h>
#include <immintrin.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Configuration
 * ═══════════════════════════════════════════════════════════════════════════ */

#define WIDTH  800
#define HEIGHT 600
#define NUM_SPHERES 5
#define NUM_RAYS (WIDTH * HEIGHT)
#define BENCHMARK_FRAMES 100

/* ═══════════════════════════════════════════════════════════════════════════
 * Data Structures
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    float cx, cy, cz, r;
    float cr, cg, cb;  /* Color */
} Sphere;

/* Ray pack for SIMD (SoA layout) */
typedef struct {
    float *ox, *oy, *oz;
    float *dx, *dy, *dz;
} RayPack;

/* ═══════════════════════════════════════════════════════════════════════════
 * External Rake Functions (from compiled .rk file)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Rake-compiled dot product */
extern __m256 dot(__m256 ax, __m256 ay, __m256 az,
                  __m256 bx, __m256 by, __m256 bz);

/* Rake-compiled ray-sphere intersection */
extern __m256 intersect_flat(__m256 ray_ox, __m256 ray_oy, __m256 ray_oz,
                             __m256 ray_dx, __m256 ray_dy, __m256 ray_dz,
                             float sphere_cx, float sphere_cy, float sphere_cz,
                             float sphere_r);

/* ═══════════════════════════════════════════════════════════════════════════
 * C Scalar Raytracer (for comparison)
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

    if (disc < 0.0f) {
        return -1.0f;
    }

    float t = (-b - sqrtf(disc)) / (2.0f * a);
    return t > 0.0f ? t : -1.0f;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * C SIMD Raytracer (AVX2, for comparison)
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

            /* Camera at origin looking in -Z */
            rays->ox[idx] = 0.0f;
            rays->oy[idx] = 0.0f;
            rays->oz[idx] = 0.0f;

            /* Compute ray direction */
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
 * Rendering Functions
 * ═══════════════════════════════════════════════════════════════════════════ */

void render_scalar(uint32_t *pixels, RayPack *rays, Sphere *spheres, int num_spheres) {
    for (int i = 0; i < NUM_RAYS; i++) {
        Vec3 origin = { rays->ox[i], rays->oy[i], rays->oz[i] };
        Vec3 dir = { rays->dx[i], rays->dy[i], rays->dz[i] };

        float closest_t = 1e30f;
        int closest_sphere = -1;

        for (int s = 0; s < num_spheres; s++) {
            float t = intersect_scalar(origin, dir, spheres[s]);
            if (t > 0.0f && t < closest_t) {
                closest_t = t;
                closest_sphere = s;
            }
        }

        if (closest_sphere >= 0) {
            /* Simple shading based on normal */
            float px = origin.x + closest_t * dir.x - spheres[closest_sphere].cx;
            float py = origin.y + closest_t * dir.y - spheres[closest_sphere].cy;
            float pz = origin.z + closest_t * dir.z - spheres[closest_sphere].cz;
            float len = sqrtf(px*px + py*py + pz*pz);
            float nx = px / len, ny = py / len, nz = pz / len;

            /* Light from upper-right */
            float light = fmaxf(0.0f, nx * 0.5f + ny * 0.7f + nz * 0.5f);
            light = 0.2f + 0.8f * light;  /* Ambient + diffuse */

            uint8_t r = (uint8_t)(spheres[closest_sphere].cr * light * 255.0f);
            uint8_t g = (uint8_t)(spheres[closest_sphere].cg * light * 255.0f);
            uint8_t b = (uint8_t)(spheres[closest_sphere].cb * light * 255.0f);

            pixels[i] = (255 << 24) | (r << 16) | (g << 8) | b;
        } else {
            /* Sky gradient */
            float t = 0.5f * (rays->dy[i] + 1.0f);
            uint8_t r = (uint8_t)((1.0f - t) * 255 + t * 128);
            uint8_t g = (uint8_t)((1.0f - t) * 255 + t * 178);
            uint8_t b = (uint8_t)(255);
            pixels[i] = (255 << 24) | (r << 16) | (g << 8) | b;
        }
    }
}

void render_simd_c(uint32_t *pixels, RayPack *rays, Sphere *spheres, int num_spheres,
                   float *t_buffer) {
    /* Initialize t_buffer to large values */
    for (int i = 0; i < NUM_RAYS; i++) {
        t_buffer[i] = 1e30f;
    }

    /* Process in chunks of 8 */
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

            /* Update closest hits */
            __m256 current = _mm256_loadu_ps(&t_buffer[i]);
            __m256 hit_mask = _mm256_and_ps(_mm256_cmp_ps(t, _mm256_setzero_ps(), _CMP_GT_OQ),
                                           _mm256_cmp_ps(t, current, _CMP_LT_OQ));
            __m256 new_t = _mm256_blendv_ps(current, t, hit_mask);
            _mm256_storeu_ps(&t_buffer[i], new_t);
        }
    }

    /* Generate pixels (simplified - just using distance for shading) */
    for (int i = 0; i < NUM_RAYS; i++) {
        if (t_buffer[i] < 1e29f) {
            float shade = 1.0f - (t_buffer[i] - 2.0f) / 10.0f;
            shade = fmaxf(0.2f, fminf(1.0f, shade));
            uint8_t c = (uint8_t)(shade * 255.0f);
            pixels[i] = (255 << 24) | (c << 16) | (c/2 << 8) | (c/4);
        } else {
            float t = 0.5f * (rays->dy[i] + 1.0f);
            uint8_t r = (uint8_t)((1.0f - t) * 255 + t * 128);
            uint8_t g = (uint8_t)((1.0f - t) * 255 + t * 178);
            pixels[i] = (255 << 24) | (r << 16) | (g << 8) | 255;
        }
    }
}

void render_rake(uint32_t *pixels, RayPack *rays, Sphere *spheres, int num_spheres,
                 float *t_buffer) {
    /* Initialize t_buffer */
    for (int i = 0; i < NUM_RAYS; i++) {
        t_buffer[i] = 1e30f;
    }

    /* Process using Rake-compiled intersection */
    for (int s = 0; s < num_spheres; s++) {
        for (int i = 0; i < NUM_RAYS; i += 8) {
            __m256 ox = _mm256_loadu_ps(&rays->ox[i]);
            __m256 oy = _mm256_loadu_ps(&rays->oy[i]);
            __m256 oz = _mm256_loadu_ps(&rays->oz[i]);
            __m256 dx = _mm256_loadu_ps(&rays->dx[i]);
            __m256 dy = _mm256_loadu_ps(&rays->dy[i]);
            __m256 dz = _mm256_loadu_ps(&rays->dz[i]);

            /* Call Rake-compiled function */
            __m256 t = intersect_flat(ox, oy, oz, dx, dy, dz,
                                      spheres[s].cx, spheres[s].cy,
                                      spheres[s].cz, spheres[s].r);

            /* Update closest hits */
            __m256 current = _mm256_loadu_ps(&t_buffer[i]);
            __m256 hit_mask = _mm256_and_ps(_mm256_cmp_ps(t, _mm256_setzero_ps(), _CMP_GT_OQ),
                                           _mm256_cmp_ps(t, current, _CMP_LT_OQ));
            __m256 new_t = _mm256_blendv_ps(current, t, hit_mask);
            _mm256_storeu_ps(&t_buffer[i], new_t);
        }
    }

    /* Generate pixels */
    for (int i = 0; i < NUM_RAYS; i++) {
        if (t_buffer[i] < 1e29f) {
            float shade = 1.0f - (t_buffer[i] - 2.0f) / 10.0f;
            shade = fmaxf(0.2f, fminf(1.0f, shade));
            uint8_t g = (uint8_t)(shade * 255.0f);
            pixels[i] = (255 << 24) | (g/4 << 16) | (g << 8) | (g/2);
        } else {
            float t = 0.5f * (rays->dy[i] + 1.0f);
            uint8_t r = (uint8_t)((1.0f - t) * 255 + t * 128);
            uint8_t g = (uint8_t)((1.0f - t) * 255 + t * 178);
            pixels[i] = (255 << 24) | (r << 16) | (g << 8) | 255;
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

int main(int argc, char *argv[]) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           Rake Raytracer Performance Comparison             ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Press 1: C Scalar    2: C SIMD (AVX2)    3: Rake SIMD      ║\n");
    printf("║  Press Q: Quit        Space: Toggle Animation               ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* Initialize SDL */
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL init failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow(
        "Rake Raytracer Demo",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WIDTH, HEIGHT,
        SDL_WINDOW_SHOWN
    );

    if (!window) {
        fprintf(stderr, "Window creation failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    SDL_Texture *texture = SDL_CreateTexture(renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        WIDTH, HEIGHT);

    /* Allocate buffers (aligned for SIMD) */
    RayPack rays;
    rays.ox = aligned_alloc(32, NUM_RAYS * sizeof(float));
    rays.oy = aligned_alloc(32, NUM_RAYS * sizeof(float));
    rays.oz = aligned_alloc(32, NUM_RAYS * sizeof(float));
    rays.dx = aligned_alloc(32, NUM_RAYS * sizeof(float));
    rays.dy = aligned_alloc(32, NUM_RAYS * sizeof(float));
    rays.dz = aligned_alloc(32, NUM_RAYS * sizeof(float));

    float *t_buffer = aligned_alloc(32, NUM_RAYS * sizeof(float));
    uint32_t *pixels = aligned_alloc(32, NUM_RAYS * sizeof(uint32_t));

    /* Generate rays */
    generate_rays(&rays, WIDTH, HEIGHT);

    /* Setup spheres */
    Sphere spheres[NUM_SPHERES] = {
        { 0.0f,  0.0f, -5.0f, 1.0f,  1.0f, 0.2f, 0.2f },  /* Red center */
        {-2.5f,  0.0f, -6.0f, 1.0f,  0.2f, 1.0f, 0.2f },  /* Green left */
        { 2.5f,  0.0f, -6.0f, 1.0f,  0.2f, 0.2f, 1.0f },  /* Blue right */
        { 0.0f,  2.0f, -7.0f, 0.8f,  1.0f, 1.0f, 0.2f },  /* Yellow top */
        { 0.0f, -2.0f, -4.0f, 0.6f,  1.0f, 0.2f, 1.0f },  /* Magenta bottom */
    };

    /* Rendering state */
    int mode = 3;  /* 1=scalar, 2=C SIMD, 3=Rake */
    bool animate = true;
    bool running = true;
    float time = 0.0f;

    /* FPS tracking */
    double fps_timer = get_time_ms();
    int frame_count = 0;
    double fps = 0.0;
    double render_time_ms = 0.0;

    /* Benchmark results */
    double bench_scalar = 0, bench_simd = 0, bench_rake = 0;
    bool benchmarked = false;

    printf("Running initial benchmark...\n");

    /* Run benchmark */
    for (int m = 1; m <= 3; m++) {
        double total = 0;
        for (int f = 0; f < BENCHMARK_FRAMES; f++) {
            double start = get_time_ms();
            switch (m) {
                case 1: render_scalar(pixels, &rays, spheres, NUM_SPHERES); break;
                case 2: render_simd_c(pixels, &rays, spheres, NUM_SPHERES, t_buffer); break;
                case 3: render_rake(pixels, &rays, spheres, NUM_SPHERES, t_buffer); break;
            }
            total += get_time_ms() - start;
        }
        double avg = total / BENCHMARK_FRAMES;
        switch (m) {
            case 1: bench_scalar = avg; break;
            case 2: bench_simd = avg; break;
            case 3: bench_rake = avg; break;
        }
    }
    benchmarked = true;

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    Benchmark Results                         ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  C Scalar:     %6.2f ms/frame  (%5.1f FPS)                   ║\n",
           bench_scalar, 1000.0/bench_scalar);
    printf("║  C SIMD:       %6.2f ms/frame  (%5.1f FPS)  %.1fx faster     ║\n",
           bench_simd, 1000.0/bench_simd, bench_scalar/bench_simd);
    printf("║  Rake SIMD:    %6.2f ms/frame  (%5.1f FPS)  %.1fx faster     ║\n",
           bench_rake, 1000.0/bench_rake, bench_scalar/bench_rake);
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* Main loop */
    SDL_Event event;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            } else if (event.type == SDL_KEYDOWN) {
                switch (event.key.keysym.sym) {
                    case SDLK_q: running = false; break;
                    case SDLK_1: mode = 1; break;
                    case SDLK_2: mode = 2; break;
                    case SDLK_3: mode = 3; break;
                    case SDLK_SPACE: animate = !animate; break;
                }
            }
        }

        /* Animate spheres */
        if (animate) {
            time += 0.016f;
            spheres[0].cx = sinf(time) * 1.5f;
            spheres[0].cy = cosf(time * 0.7f) * 0.5f;
            spheres[1].cy = sinf(time * 1.3f) * 1.0f;
            spheres[2].cy = cosf(time * 1.1f) * 1.0f;
            spheres[3].cx = cosf(time * 0.8f) * 2.0f;
            spheres[4].cz = -4.0f + sinf(time * 0.9f) * 1.0f;
        }

        /* Render */
        double render_start = get_time_ms();
        switch (mode) {
            case 1: render_scalar(pixels, &rays, spheres, NUM_SPHERES); break;
            case 2: render_simd_c(pixels, &rays, spheres, NUM_SPHERES, t_buffer); break;
            case 3: render_rake(pixels, &rays, spheres, NUM_SPHERES, t_buffer); break;
        }
        render_time_ms = get_time_ms() - render_start;

        /* Update texture */
        SDL_UpdateTexture(texture, NULL, pixels, WIDTH * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);

        /* FPS calculation */
        frame_count++;
        double now = get_time_ms();
        if (now - fps_timer >= 1000.0) {
            fps = frame_count * 1000.0 / (now - fps_timer);
            frame_count = 0;
            fps_timer = now;

            const char *mode_name = mode == 1 ? "C Scalar" :
                                    mode == 2 ? "C SIMD" : "Rake SIMD";
            char title[256];
            snprintf(title, sizeof(title),
                     "Rake Demo | %s | %.1f FPS | %.2f ms/frame | %.1fx speedup vs scalar",
                     mode_name, fps, render_time_ms, bench_scalar / render_time_ms);
            SDL_SetWindowTitle(window, title);
        }
    }

    /* Cleanup */
    free(rays.ox); free(rays.oy); free(rays.oz);
    free(rays.dx); free(rays.dy); free(rays.dz);
    free(t_buffer);
    free(pixels);

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
