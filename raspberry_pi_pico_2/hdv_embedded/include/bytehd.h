#ifndef BYTEHD_H
#define BYTEHD_H

#include <stdint.h>
#include "hdc.h"

#define NUMBERS_IN_A_BYTE 8
#define LENGTH_HDV_BYTES ((HDV_DIM + NUMBERS_IN_A_BYTE - 1) / NUMBERS_IN_A_BYTE)

void gen_random_encoded_hdv(char hdv[LENGTH_HDV_BYTES]);
void encode_array_to_hdv(int8_t values[HDV_DIM], char hdv_bytes [LENGTH_HDV_BYTES]);

// ***********************************************************************************************************
// ******************** THE FOLLOWING FUNCTIONS ARE SUPPOSED TO BE USED WITH ENCODED HDVs ********************
// ***********************************************************************************************************

void decode_hdv_to_array(char hdv_bytes[LENGTH_HDV_BYTES], int8_t values[HDV_DIM]);
uint8_t get_number_from_hdv(char hdv_bytes[LENGTH_HDV_BYTES], uint16_t index);
int8_t set_number_in_hdv(char hdv_bytes[LENGTH_HDV_BYTES], uint16_t index, int8_t value);
void print_hdv(char hdv_bytes[LENGTH_HDV_BYTES]);


#endif