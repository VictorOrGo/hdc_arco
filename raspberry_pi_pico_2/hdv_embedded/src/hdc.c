#include <math.h>
// #include <pico/rand.h>
// #include <pico/stdlib.h>
#include <stdlib.h>
#include "hdc.h"

void gen_random_bipolar_hdv(int8_t hdv[HDV_DIM]) {
    for (int i = 0; i < HDV_DIM; i++) {
        hdv[i] = (rand() % 2) * 2 - 1; // Randomly generate -1 or 1
    }
}

int8_t ensure_hdv_bipolar(int8_t hdv[HDV_DIM]) {
    for (int i = 0; i < HDV_DIM; i++) {
        if (hdv[i] != 1 && hdv[i] != -1) {
            return 1; // Found a value that is not -1 or 1
        }
    }
    return 0; // All values are either -1 or 1
}

void normalize_bipolar_hdv(int8_t hdv[HDV_DIM]) {
    for (int i = 0; i < HDV_DIM; i++) {
        hdv[i] = (hdv[i] > 0) ? 1 : -1;
    }
}

void bundle_hdv(int8_t hdv1[HDV_DIM], int8_t hdv2[HDV_DIM], int8_t result[HDV_DIM]) {
    for (int i = 0; i < HDV_DIM; i++) {
        result[i] = hdv1[i] + hdv2[i];
    }
}

void bind_hdv(int8_t hdv1[HDV_DIM], int8_t hdv2[HDV_DIM], int8_t result[HDV_DIM]) {
    for (int i = 0; i < HDV_DIM; i++) {
        result[i] = hdv1[i] * hdv2[i];
    }
}

void permute_hdv(int8_t hdv[HDV_DIM], int8_t result[HDV_DIM], int shift) {
    for (int i = 0; i < HDV_DIM; i++) {
        int new_index = (i + shift + HDV_DIM) % HDV_DIM; // ensure positive index with + HDV_DIM before modulo
        result[new_index] = hdv[i];
    }
}

float cosine_similarity(int8_t hdv1[HDV_DIM], int8_t hdv2[HDV_DIM]) {
    float dot = 0.0, mag1 = 0.0, mag2 = 0.0;
    for (int i = 0; i < HDV_DIM; i++) {
        dot += hdv1[i] * hdv2[i];
        mag1 += hdv1[i] * hdv1[i];
        mag2 += hdv2[i] * hdv2[i];
    }
    if (mag1 == 0 || mag2 == 0) return 0.0;
    return dot / (sqrt(mag1) * sqrt(mag2));
}