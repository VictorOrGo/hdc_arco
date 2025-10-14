#ifndef HDC_H
#define HDC_H

#include <stdint.h>

#define HDV_DIM 10000

void gen_random_bipolar_hdv(int8_t hdv[HDV_DIM]);
int8_t ensure_hdv_bipolar(int8_t hdv[HDV_DIM]);
void normalize_bipolar_hdv(int8_t hdv[HDV_DIM]);
void bundle_hdv(int8_t hdv1[HDV_DIM], int8_t hdv2[HDV_DIM], int8_t result[HDV_DIM]);
void bind_hdv(int8_t hdv1[HDV_DIM], int8_t hdv2[HDV_DIM], int8_t result[HDV_DIM]);
void permute_hdv(int8_t hdv[HDV_DIM], int8_t result[HDV_DIM], int shift);
float cosine_similarity(int8_t hdv1[HDV_DIM], int8_t hdv2[HDV_DIM]);

#endif