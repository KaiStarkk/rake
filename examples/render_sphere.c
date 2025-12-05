// Simple sphere renderer using Rake-generated intersection code
// Outputs a PPM image file

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

// Rake-generated function (from intersect_flat.rk)
extern __m256 intersect_flat(
    __m256 ray_ox, __m256 ray_oy, __m256 ray_oz,
    __m256 ray_dx, __m256 ray_dy, __m256 ray_dz,
    float sphere_cx, float sphere_cy, float sphere_cz, float sphere_r
);

#define WIDTH 800
#define HEIGHT 600

int main() {
    // Image buffer
    unsigned char *image = malloc(WIDTH * HEIGHT * 3);

    // Camera setup
    float cam_x = 0.0f, cam_y = 0.0f, cam_z = 5.0f;
    float fov = 1.0f;  // tan(fov/2)
    float aspect = (float)WIDTH / HEIGHT;

    // Sphere at origin
    float sphere_cx = 0.0f, sphere_cy = 0.0f, sphere_cz = 0.0f;
    float sphere_r = 1.5f;

    // Light direction (normalized)
    float light_x = 0.5f, light_y = 0.7f, light_z = 0.5f;
    float light_len = sqrtf(light_x*light_x + light_y*light_y + light_z*light_z);
    light_x /= light_len; light_y /= light_len; light_z /= light_len;

    // Render in batches of 8 pixels (SIMD width)
    float ray_dx[8] __attribute__((aligned(32)));
    float ray_dy[8] __attribute__((aligned(32)));
    float ray_dz[8] __attribute__((aligned(32)));
    float t_values[8] __attribute__((aligned(32)));

    printf("Rendering %dx%d sphere...\n", WIDTH, HEIGHT);

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x += 8) {
            // Generate 8 rays for this batch
            for (int i = 0; i < 8 && (x + i) < WIDTH; i++) {
                float px = (x + i);
                float py = y;

                // Convert pixel to normalized device coordinates
                float ndc_x = (2.0f * px / WIDTH - 1.0f) * aspect * fov;
                float ndc_y = (1.0f - 2.0f * py / HEIGHT) * fov;

                // Ray direction (from camera through pixel)
                ray_dx[i] = ndc_x;
                ray_dy[i] = ndc_y;
                ray_dz[i] = -1.0f;

                // Normalize
                float len = sqrtf(ray_dx[i]*ray_dx[i] + ray_dy[i]*ray_dy[i] + ray_dz[i]*ray_dz[i]);
                ray_dx[i] /= len;
                ray_dy[i] /= len;
                ray_dz[i] /= len;
            }

            // Load rays into SIMD registers
            __m256 ox = _mm256_set1_ps(cam_x);
            __m256 oy = _mm256_set1_ps(cam_y);
            __m256 oz = _mm256_set1_ps(cam_z);
            __m256 dx = _mm256_load_ps(ray_dx);
            __m256 dy = _mm256_load_ps(ray_dy);
            __m256 dz = _mm256_load_ps(ray_dz);

            // Intersect!
            __m256 t = intersect_flat(ox, oy, oz, dx, dy, dz,
                sphere_cx, sphere_cy, sphere_cz, sphere_r);

            _mm256_store_ps(t_values, t);

            // Shade each pixel
            for (int i = 0; i < 8 && (x + i) < WIDTH; i++) {
                int idx = (y * WIDTH + x + i) * 3;

                if (t_values[i] > 0.0f) {
                    // Hit! Compute hit point and normal
                    float hit_x = cam_x + t_values[i] * ray_dx[i];
                    float hit_y = cam_y + t_values[i] * ray_dy[i];
                    float hit_z = cam_z + t_values[i] * ray_dz[i];

                    // Normal (sphere centered at origin)
                    float nx = (hit_x - sphere_cx) / sphere_r;
                    float ny = (hit_y - sphere_cy) / sphere_r;
                    float nz = (hit_z - sphere_cz) / sphere_r;

                    // Diffuse lighting
                    float ndotl = nx * light_x + ny * light_y + nz * light_z;
                    if (ndotl < 0.0f) ndotl = 0.0f;

                    // Ambient + diffuse
                    float intensity = 0.15f + 0.85f * ndotl;

                    // Desert sand color (matching website theme)
                    image[idx + 0] = (unsigned char)(224 * intensity);  // R
                    image[idx + 1] = (unsigned char)(176 * intensity);  // G
                    image[idx + 2] = (unsigned char)(128 * intensity);  // B
                } else {
                    // Miss - dark background gradient
                    float gradient = 1.0f - (float)y / HEIGHT;
                    image[idx + 0] = (unsigned char)(30 + 20 * gradient);   // R
                    image[idx + 1] = (unsigned char)(20 + 16 * gradient);   // G
                    image[idx + 2] = (unsigned char)(16 + 12 * gradient);   // B
                }
            }
        }
    }

    // Write PPM file
    FILE *fp = fopen("sphere.ppm", "wb");
    fprintf(fp, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    fwrite(image, 1, WIDTH * HEIGHT * 3, fp);
    fclose(fp);

    printf("Wrote sphere.ppm\n");

    free(image);
    return 0;
}
