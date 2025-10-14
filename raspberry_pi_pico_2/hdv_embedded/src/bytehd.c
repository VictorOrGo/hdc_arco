#include "bytehd.h"
#include "hdc.h"
// #include <pico/rand.h>
// #include <pico/stdlib.h>
#include <stdlib.h>
#include <stdio.h>
#define false 0
#define true 1

int8_t DECODE_MAP[2] = {-1, 1}; // Map 0 to -1 and 1 to 1

void gen_random_encoded_hdv(char hdv[LENGTH_HDV_BYTES]) {

    for (int i = 0; i < (LENGTH_HDV_BYTES / 8) + 1; i++) {
        hdv[i] = (rand() % 255) & 0xFF; // Random number between 0 and 255 to store a random byte
    }

    int remaining_bits = (LENGTH_HDV_BYTES * 8) - HDV_DIM;
    if (remaining_bits > 0) {
        unsigned char mask = 0xFF << remaining_bits; // Shift left to create the mask that will turn the remaining bits to 0
        hdv[LENGTH_HDV_BYTES - 1] &= mask;
    }
}


void encode_array_to_hdv(int8_t values[HDV_DIM], char hdv_bytes [LENGTH_HDV_BYTES]) {

    if (ensure_hdv_bipolar(values) == false) {
        printf("Error: HDV values must be either -1 or 1.\n");
        return;
    }

    for (int i = 0; i < LENGTH_HDV_BYTES; i++) { // we clean the bytes
        hdv_bytes[i] = 0x00; 
    }

    for (int i = 0; i < HDV_DIM; i++) {
        int byte_index = i / NUMBERS_IN_A_BYTE;
        int8_t bit_index  = 7 - (i % NUMBERS_IN_A_BYTE); // MSB first

        int8_t bit = 0;
        if (values[i] == 1) { // We map 1 to bit 1 and -1 to bit 0
            bit = 1;
        } else { 
            bit = 0;
        }

        hdv_bytes[byte_index] |= (bit << bit_index); // we shift the bit to its position and set it in the byte using OR
    }
}

void decode_hdv_to_array(char hdv_bytes[LENGTH_HDV_BYTES], int8_t values[HDV_DIM]) {
    for (int i = 0; i < LENGTH_HDV_BYTES; i++) { // for each byte

        int8_t bits_to_decode = 0;
        if (HDV_DIM - i * 8 >= 8) { // hdv_dim - the number of bits already decoded. if it is higher than 8, we decode 8 bits (next byte)
            bits_to_decode = 8;
        } else {
            bits_to_decode = HDV_DIM - i * 8;
        }

        for (int j = 0; j < bits_to_decode; j++) { // extraction of each bit msb first
            int8_t bit = (hdv_bytes[i] >> (7 - j)) & 0x01; // we extract the bit (0 or 1) by shifting it to position 0 and applying a mask
            values[i * 8 + j] = DECODE_MAP[bit];        // we convert it to -1 or 1
        }
    }
}

uint8_t get_number_from_hdv(char hdv_bytes[LENGTH_HDV_BYTES], uint16_t index) {
    if (index < 0 || index >= HDV_DIM) {
        printf("Error: Index out of bounds.\n");
        return 2;
    }

    uint16_t byte_index = index / NUMBERS_IN_A_BYTE; // We find the byte that contains the number of the selected index
    uint8_t bit_index = 7 - (index % NUMBERS_IN_A_BYTE); // MSB first

    int8_t bit = (hdv_bytes[byte_index] >> bit_index) & 0x01; // Extract the bit at the specified index
    return DECODE_MAP[bit]; // Map 0 to -1 and 1 to 1
}

int8_t set_number_in_hdv(char hdv_bytes[LENGTH_HDV_BYTES], uint16_t index, int8_t value) {
    if (index < 0 || index >= HDV_DIM) {
        printf("Error: Index out of bounds.\n");
        return -1;
    }
    if (value != -1 && value != 1) {
        printf("Error: Value must be either -1 or 1.\n");
        return -1;
    }

    uint16_t byte_index = index / NUMBERS_IN_A_BYTE; // We find the byte that contains the number of the selected index
    uint8_t bit_index = 7 - (index % NUMBERS_IN_A_BYTE); // MSB first

    if (value == 1) {
        hdv_bytes[byte_index] |= (1 << bit_index); // Set the bit to 1 using OR
    } else { 
        hdv_bytes[byte_index] &= ~(1 << bit_index); // Set the bit to 0 using AND with NOT
    }

    return 0; 
}

void print_hdv(char hdv_bytes[LENGTH_HDV_BYTES]) {
    printf("[");
    for (int i = 0; i < HDV_DIM; i++) {
        int8_t num = get_number_from_hdv(hdv_bytes, i);
        if (num == 2) {
            printf("Error: get_number_from_hdv returned error value 2.\n");
            return;
        }
        printf("%d, ", num);
    }
    printf("]\n");
}