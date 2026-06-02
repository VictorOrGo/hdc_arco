#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include "pico/stdlib.h" // Includes sleep_ms()
#include "pico/stdio_usb.h"
#include <math.h>
#include <time.h>
#include <string.h>
#include <malloc.h>
#include "hdc.h"
#include "bytehd_lib.h"
#include <limits.h>

// D_values = [1000, 1500, 2000, 2500, 3000, 3500, 4000]
// M_values = [74,    54,   39,   26,   21,   16,   13]

#define M 54
#define NORMALIZATION_SUM 100
#define FEATURES 32

typedef struct {
   float travel_ms;
   float hops;
   float hours_in_emergency;
   float hours_in_power;
   float times_in_emergency;
   float times_in_power;
   float battery_level;
   float link_quality;
   float WBN_rssi_correction_val;
   float cbmac_blacklisting_channels_min_to_40;
   float cluster_channel;
   float scanstat_avg_routers;
   float network_scans_amount;
   float trace_options_sequence;
   float cbmac_details_cbmac_load;
   float cbmac_details_cbmac_rx_messages_ack;
   float cbmac_details_cbmac_rx_messages_unack;
   float cbmac_details_cbmac_rx_ack_other_reasons;
   float cbmac_details_cbmac_tx_ack_cca_fail;
   float cbmac_details_cbmac_tx_ack_not_received;
   float cbmac_details_cbmac_tx_messages_ack;
   float cbmac_details_cbmac_tx_messages_unack;
   float cbmac_details_cbmac_tx_cca_unack_fail;
   float buffer_usage_average;
   float nexthop_details_advertised_cost;
   float nexthop_details_next_hop_quality;
   float nexthop_details_next_hop_rssi;
   float nexthop_details_next_hop_power;
   float installation_quality_quality_indicator;
   float nexthop_details_next_hop_address;
   float cfmac_pending_broadcast_le_member;
   float unack_broadcast_channel;
   char classification;
} json_data_t;

typedef struct {
    const char *name;  // nombre de la variable
    int lower;
    int higher;
} range_t;

uint8_t range_hdv_levels(char **matrix_f, int num_features, int m_levels){
    
    gen_random_encoded_hdv(matrix_f[0]);

    int b = HDV_DIM / (2 * (m_levels - 1));
    if (b == 0) {
        b = 1;
    }

    // if (b * (m_levels - 1) > HDV_DIM) {
    //     printf("Error: Not enough unique indices for the number b = %d.\n", b);
    //     return 1;
    // }

    bool used_indices[HDV_DIM] = {0}; // example: indice = 72 -> used_indices[72] = true 
    int indices_selected[b];
    int used_indices_count = 0;

    for (int i = 1; i < m_levels; i++) {
        if (used_indices_count + b > HDV_DIM) { // Reset if we run out of unique indices
            memset(used_indices, false, sizeof(used_indices));
            used_indices_count = 0;
        }

        memcpy(matrix_f[i], matrix_f[i - 1], LENGTH_HDV_BYTES * sizeof(char)); // Copy previous HDV to the new HDV

        // *********************************************************************
        // ****************** SELECTION OF B UNIQUE INICES *********************
        // *********************************************************************

        uint16_t rand_indice;
        for (int j = 0; j < b; j++) {
            do {
                rand_indice = my_rand() % HDV_DIM;
            } while (used_indices[rand_indice]);
            
            used_indices[rand_indice] = true;
            indices_selected[j] = rand_indice;
            used_indices_count++;
        }

        // *********************************************************************
        // ****************** GENERATION OF THE NEW HDV ***********************
        // *********************************************************************

        for (int k = 0; k < b; k++) { // For each selected index, flip the bit
            uint8_t num = get_number_from_hdv(matrix_f[i], indices_selected[k]);
            if (num == 2) {
                printf("Error: get_number_from_hdv returned error value 2.\n");
                return 1;
            }
        
            else if(num == 1) {
                if (set_number_in_hdv(matrix_f[i], indices_selected[k], -1) == -1) {
                    printf("Error: set_number_in_hdv returned error value -1.\n");
                    return 1;
                }
            }
            else {
                if (set_number_in_hdv(matrix_f[i], indices_selected[k], 1) == -1) {
                    printf("Error: set_number_in_hdv returned error value -1.\n");
                    return 1;
                }
            }
        }
        
    }

    return 0;
}

int8_t get_hdv_level(char **matrix_f, char *result_hdv, float lower, float higher, float value, uint16_t m_levels){

    float temp = (fabs(higher) + fabs(lower)) / m_levels; // Range divided by number of levels
    float increment = round(temp * 10000) / 10000; // Round to 4 decimal places
    uint16_t index = (uint16_t) round((value - lower) / increment);

    if (index < 0 || index >= m_levels) {
        return -1;
    }

    memcpy(result_hdv, matrix_f[index], LENGTH_HDV_BYTES * sizeof(char)); // Copy the corresponding HDV to result_hdv
    return 0;
}

int main() {
    my_srand(500);

    stdio_init_all();
    sleep_ms(1000); // Wait for USB to be ready

    range_t ranges[] = {
    {"travel_ms", -11.129908607600001, 10.3422309835},
    {"hops", -10.5285593448, 9.9666576071},
    {"hours_in_emergency", -9.2021338146, 10.7978661854},
    {"hours_in_power", -10.705300119, 10.5583666203},
    {"times_in_emergency", -8.9597496678, 11.0402503322},
    {"times_in_power", -9.8195573402, 10.1804426598},
    {"battery_level", -9.3706225416, 9.8756683545},
    {"link_quality", -9.266326185, 10.733673815},
    {"WBN_rssi_correction_val", -11.7412827994, 10.9541171806},
    {"cbmac_blacklisting_channels_min_to_40", -10.2810097094, 9.681431759},
    {"cluster_channel", -9.6426633002, 10.8128270088},
    {"scanstat_avg_routers", -10.0606091916, 9.9393908084},
    {"network_scans_amount", -8.8622114835, 11.4624569702},
    {"trace_options_sequence", -9.4283203761, 10.7883938603},
    {"cbmac_details_cbmac_load", -11.3532607214, 9.8509129535},
    {"cbmac_details_cbmac_rx_messages_ack", -11.6431392286, 10.8331624216},
    {"cbmac_details_cbmac_rx_messages_unack", -8.2238746191, 9.2849584411},
    {"cbmac_details_cbmac_rx_ack_other_reasons", -10.7392980779, 9.2607019221},
    {"cbmac_details_cbmac_tx_ack_cca_fail", -9.3067637164, 10.8998948603},
    {"cbmac_details_cbmac_tx_ack_not_received", -8.4917809143, 9.4613816643},
    {"cbmac_details_cbmac_tx_messages_ack", -10.920853884, 10.2657952991},
    {"cbmac_details_cbmac_tx_messages_unack", -8.7135986302, 11.635152705},
    {"cbmac_details_cbmac_tx_cca_unack_fail", -8.9506993171, 11.0521759556},
    {"buffer_usage_average", -12.7949269663, 10.3570059096},
    {"nexthop_details_advertised_cost", -10.2220550246, 10.3007748668},
    {"nexthop_details_next_hop_quality", -8.8687137216, 9.6393607259},
    {"nexthop_details_next_hop_rssi", -9.583995872, 9.4240177678},
    {"nexthop_details_next_hop_power", -9.4284614183, 10.5715385817},
    {"installation_quality_quality_indicator", -9.4833139316, 10.5166860684},
    {"nexthop_details_next_hop_address", 5, 203},
    {"cfmac_pending_broadcast_le_member", 12, 34},
    {"unack_broadcast_channel", 6, 21}
    };

    // *********************************************************************
    // ********************* ALLOCATE MEMORY FOR MATRICES ******************
    // *********************************************************************

    char ***vector_matrices; 
    // vector_matrices[i] points to the matrix i (feature i)
    // vector_matrices[i][j] points to the hypervector j (level hdv) of the matrix i (feature i)
    // vector_matrices[i][j][k] points to the component k (component k) of the hypervector j (level hdv) of the matrix i (feature i)
    // char vector_matrices[FEATURES][M][LENGTH_HDV_BYTES]; 

    printf("Allocating memory... \n");

    vector_matrices = malloc(FEATURES * sizeof(char**)); // Allocate memory for array of pointers to 2D matrices, to every feature matrix
    if (vector_matrices == NULL) {
        return 1;
    }
    for(int i = 0; i < FEATURES; i++) { 
        vector_matrices[i] = malloc(M * sizeof(char*)); // Allocate memory for M rows, each level HDV
        if (vector_matrices[i] == NULL) {
            return 1;
        }
        for(int j = 0; j < M; j++) {
            vector_matrices[i][j] = malloc(LENGTH_HDV_BYTES * sizeof(char)); // Allocate memory for each component of the HDV
            if (vector_matrices[i][j] == NULL) {
                return 1;
            }
        }
    }

    printf("Memory allocated succesfully. \n");
    
    printf("Initializing allocated memory... \n");
    for(int i = 0; i < FEATURES; i++) { 
        for(int j = 0; j < M; j++) {
            for(int k = 0; k < LENGTH_HDV_BYTES; k++) {
                vector_matrices[i][j][k] = 0; // Initialize all components to 0
            }
        }
    }
    printf("Memory initialized succesfully. \n");

    // *********************************************************************
    // ********************* GENERATE LEVEL HDVs ***************************
    // *********************************************************************

    printf("Creating HDV levels... \n");
    for(int i = 0; i < FEATURES; i++) {  // For each feature we create the level HDVs
        if(range_hdv_levels(vector_matrices[i], FEATURES, M) != 0) {
            printf("Error: range_hdv_levels returned error value.\n");
            return 1;
        }
    }

    printf("HDV levels created succesfully. \n");

    // int8_t hdv0[HDV_DIM];
    // printf("\n");
    // decode_hdv_to_array(vector_matrices[0][0], hdv0);
    // for (int i = 0; i < M; i++) { 
    //     int8_t hdv_i[HDV_DIM];
    //     decode_hdv_to_array(vector_matrices[0][i], hdv_i);
    //     float cos = cosine_similarity(hdv0, hdv_i);
    //     printf("Cosine similarity of vector %d with level 0: %f \n", i, cos);
    // }

    int16_t prot_hdv[HDV_DIM];
    int16_t sample_hdv[HDV_DIM];
    char aux_hdv[LENGTH_HDV_BYTES];
    int16_t aux_hdv_decoded[HDV_DIM];
    char *line = NULL;
    int num_line = 0; 
    int num_lines_skipped = 0;
    int8_t bundle_count = 0;
    size_t len = 0;
    json_data_t data;
    int lower;
    int higher;
    float *ptr = &data.travel_ms; // Pointer to the first field of the struct int *ptr = (int *)&data; 
    float value;
    clock_t start = clock();
    clock_t end = 0;
    double elapsed = 0;

    printf("Connecting to receive data...\n");

    while (!stdio_usb_connected()) {
        sleep_ms(100);  // Espera a que el USB esté listo (opcional)
    }

    printf("Connected\n");

    // **************************************************************
    // ******************** RECEIVE THRESHOLD ***********************
    // **************************************************************

    float threshold = 0;
    char buffer_threshold[64];
    int pos = 0;

    while (true) {
        int c = getchar_timeout_us(0);
        if (c != PICO_ERROR_TIMEOUT) {
            if (c == '\n' || c == '\r') {
                buffer_threshold[pos] = '\0';  // Termina la cadena

                if (threshold != 0) {
                    break; 
                }

                // Convertimos el buffer a float
                threshold = strtof(buffer_threshold, NULL);

                // Para debug: imprime el valor recibido
                printf("Threshold recibido: %f\n", threshold);

                pos = 0;  // Reset para la siguiente línea
            } 
            else if (pos < sizeof(buffer_threshold) - 1) {
                buffer_threshold[pos++] = (char)c;
            }
        }
        sleep_ms(1);
    }

    // ************************************************
    // ******************* TRAINING *******************
    // ************************************************

    printf("Starting training...\n");

    char buffer[512];
    pos = 0;
    value = 0;

    while (true) {
        int c = getchar_timeout_us(0);
        if (c != PICO_ERROR_TIMEOUT) {
            if (c == '\n' || c == '\r') {
                buffer[pos] = '\0';

                if (strcmp(buffer, "stop") == 0) {
                    break; 
                }

                if (pos > 0) {
                    float values[FEATURES];
                    int parsed = 0;
                    char *token = strtok(buffer, ",");

                    while (token && parsed < FEATURES) {
                        values[parsed++] = atof(token);
                        token = strtok(NULL, ",");
                    }

                    if (parsed == FEATURES) {
                        data.travel_ms = values[0];
                        data.hops = values[1];
                        data.hours_in_emergency = values[2];
                        data.hours_in_power = values[3];
                        data.times_in_emergency = values[4];
                        data.times_in_power = values[5];
                        data.battery_level = values[6];
                        data.link_quality = values[7];
                        data.WBN_rssi_correction_val = values[8];
                        data.cbmac_blacklisting_channels_min_to_40 = values[9];
                        data.cluster_channel = values[10];
                        data.scanstat_avg_routers = values[11];
                        data.network_scans_amount = values[12];
                        data.trace_options_sequence = values[13];
                        data.cbmac_details_cbmac_load = values[14];
                        data.cbmac_details_cbmac_rx_messages_ack = values[15];
                        data.cbmac_details_cbmac_rx_messages_unack = values[16];
                        data.cbmac_details_cbmac_rx_ack_other_reasons = values[17];
                        data.cbmac_details_cbmac_tx_ack_cca_fail = values[18];
                        data.cbmac_details_cbmac_tx_ack_not_received = values[19];
                        data.cbmac_details_cbmac_tx_messages_ack = values[20];
                        data.cbmac_details_cbmac_tx_messages_unack = values[21];
                        data.cbmac_details_cbmac_tx_cca_unack_fail = values[22];
                        data.buffer_usage_average = values[23];
                        data.nexthop_details_advertised_cost = values[24];
                        data.nexthop_details_next_hop_quality = values[25];
                        data.nexthop_details_next_hop_rssi = values[26];
                        data.nexthop_details_next_hop_power = values[27];
                        data.installation_quality_quality_indicator = values[28];
                        data.nexthop_details_next_hop_address = values[29];
                        data.cfmac_pending_broadcast_le_member = values[30];
                        data.unack_broadcast_channel = values[31];

                        // *************************************************
                        // *************** PROCESS THE DATA *****************
                        // *************************************************

                        for (int i = 0; i < FEATURES; i++) { // Creation of the sample HDV by bundling the HDVs of each feature
                            value =  ptr[i]; // get the value of the field

                            if (value == INT_MIN) { // Check for Null value in json
                                continue;
                            }
                    
                            lower = ranges[i].lower;
                            higher = ranges[i].higher;
                            get_hdv_level(vector_matrices[i], aux_hdv, lower, higher, value, M);

                            decode_hdv_to_array(aux_hdv, aux_hdv_decoded); // Decode the HDV from bytes to int16_t array
                            
                            if (i == 0) {
                                memcpy(sample_hdv, aux_hdv_decoded, HDV_DIM * sizeof(int16_t)); // Initialize sample_hdv with the first feature HDV
                                continue;
                            }
                            bundle_hdv(sample_hdv, aux_hdv_decoded, sample_hdv); // Accumulate the HDVs of each feature
                        }

                        normalize_bipolar_hdv(sample_hdv); 

                        if(num_line == 0) {
                            memcpy(prot_hdv, sample_hdv, HDV_DIM * sizeof(int16_t)); // Initialize prototype HDV with the first sample HDV
                        }
                        else if (bundle_count >= NORMALIZATION_SUM) {
                            bundle_hdv(prot_hdv, sample_hdv, prot_hdv); // Accumulate the sample HDVs to create the prototype HDV
                            normalize_bipolar_hdv(prot_hdv); // Normalize the prototype HDV
                            bundle_count = 0;
                        }
                        else {
                            bundle_hdv(prot_hdv, sample_hdv, prot_hdv); // Accumulate the sample HDVs to create the prototype HDV
                            bundle_count++;
                        }

                        num_line++;
                        memset(sample_hdv, 0, sizeof(sample_hdv));
                        
                    } else {
                        printf("Error: expected %d values, received %d\n",FEATURES, parsed);
                        printf("Received line: %s\n", buffer);
                    }
                }
                pos = 0;
            } 
            else if (pos < sizeof(buffer) - 1) {
                buffer[pos++] = (char)c;
            }
        }
        sleep_ms(1);
    }

    normalize_bipolar_hdv(prot_hdv); // Final normalization of the prototype HDV

    printf("Training completed. \n");
    end = clock();
    elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Training time: %f (s)\n", elapsed);

    // **************************************************************
    // ********************** TESTING *******************************
    // **************************************************************

    printf("Starting testing...\n");
    float cos_similirity = 0;
    num_line = 0;
    len = 0;
    pos = 0;
    int tp = 0;
    int tn = 0;
    int total = 0;
    int real_entry = 0;
    int synthetic_entry = 0;
    char real_classification = '\0';
    memset(sample_hdv, 0, sizeof(sample_hdv));

    while (true) {
        int c = getchar_timeout_us(0);
        if (c != PICO_ERROR_TIMEOUT) {
            if (c == '\n' || c == '\r') {
                buffer[pos] = '\0';

                if (strcmp(buffer, "stop") == 0) {
                    break; 
                }

                if (pos > 0) {
                    float values[FEATURES];
                    int parsed = 0;
                    char *token = strtok(buffer, ",");

                    while (token && parsed < FEATURES) {
                        values[parsed++] = atof(token);
                        token = strtok(NULL, ",");
                    }

                    if (parsed == FEATURES) {
                        data.travel_ms = values[0];
                        data.hops = values[1];
                        data.hours_in_emergency = values[2];
                        data.hours_in_power = values[3];
                        data.times_in_emergency = values[4];
                        data.times_in_power = values[5];
                        data.battery_level = values[6];
                        data.link_quality = values[7];
                        data.WBN_rssi_correction_val = values[8];
                        data.cbmac_blacklisting_channels_min_to_40 = values[9];
                        data.cluster_channel = values[10];
                        data.scanstat_avg_routers = values[11];
                        data.network_scans_amount = values[12];
                        data.trace_options_sequence = values[13];
                        data.cbmac_details_cbmac_load = values[14];
                        data.cbmac_details_cbmac_rx_messages_ack = values[15];
                        data.cbmac_details_cbmac_rx_messages_unack = values[16];
                        data.cbmac_details_cbmac_rx_ack_other_reasons = values[17];
                        data.cbmac_details_cbmac_tx_ack_cca_fail = values[18];
                        data.cbmac_details_cbmac_tx_ack_not_received = values[19];
                        data.cbmac_details_cbmac_tx_messages_ack = values[20];
                        data.cbmac_details_cbmac_tx_messages_unack = values[21];
                        data.cbmac_details_cbmac_tx_cca_unack_fail = values[22];
                        data.buffer_usage_average = values[23];
                        data.nexthop_details_advertised_cost = values[24];
                        data.nexthop_details_next_hop_quality = values[25];
                        data.nexthop_details_next_hop_rssi = values[26];
                        data.nexthop_details_next_hop_power = values[27];
                        data.installation_quality_quality_indicator = values[28];
                        data.nexthop_details_next_hop_address = values[29];
                        data.cfmac_pending_broadcast_le_member = values[30];
                        data.unack_broadcast_channel = values[31];
                        data.classification = token[0];

                        // *************************************************
                        // *************** PROCESS THE DATA *****************
                        // *************************************************

                        for (int i = 0; i < FEATURES + 1; i++) { // FEATURES + 1 to include the classification field
                            value =  ptr[i]; // get the value of the field
                            
                            if (i == FEATURES) { // Last field is classification
                                if (data.classification == 'r') {
                                    real_classification = 'r';
                                    real_entry++;
                                } else {
                                    real_classification = 's';
                                    synthetic_entry++;
                                }
                                break;
                            }
                    
                            lower = ranges[i].lower;
                            higher = ranges[i].higher;
                            get_hdv_level(vector_matrices[i], aux_hdv, lower, higher, value, M);

                            decode_hdv_to_array(aux_hdv, aux_hdv_decoded); // Decode the HDV from bytes to int16_t array
                            
                            if (i == 0) {
                                memcpy(sample_hdv, aux_hdv_decoded, HDV_DIM * sizeof(int16_t)); // Initialize sample_hdv with the first feature HDV
                                continue;
                            }
                            bundle_hdv(sample_hdv, aux_hdv_decoded, sample_hdv); // Accumulate the HDVs of each feature
                        }

                        normalize_bipolar_hdv(sample_hdv); 

                        num_line++;

                        cos_similirity = cosine_similarity(prot_hdv, sample_hdv); // Calculate cosine similarity between prototype HDV and sample HDV
                        
                        if (cos_similirity >= threshold && real_classification == 'r') {
                            tp++;
                        }
                        else if (cos_similirity < threshold && real_classification == 's') {
                            tn++;
                        }
                
                        total++;
                        memset(sample_hdv, 0, sizeof(sample_hdv));
                        
                    } else {
                        printf("Error: expected %d values, received %d\n", FEATURES, parsed);
                        printf("Parsed tokens: %d\n", parsed);
                        printf("Received line: %s\n", buffer);
                    }
                }
                pos = 0;
            } 
            else if (pos < sizeof(buffer) - 1) {
                buffer[pos++] = (char)c;
            }
        }
        sleep_ms(1);
    }    

    end = clock();
    elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Testing time: %f (s)\n", elapsed);

    size_t total_mem = FEATURES * sizeof(char**) + FEATURES * M * sizeof(char*) + FEATURES * M * LENGTH_HDV_BYTES * sizeof(char);
    printf("Memory matrix: %zu bytes (%ld KB, %ld MB)\n", total_mem, total_mem / 1024, total_mem / (1024 * 1024));

    int fp_m = synthetic_entry - tn;
    int fn = real_entry - tp;

    float accuracy = (float)(tp + tn) / (float)(tp + tn + fp_m + fn);
    float tpr = (float)tp / (float)(tp + fn); // True Positive Rate (Sensitivity)
    float tnr = (float)tn / (float)(tn + fp_m); // True Negative Rate (Specificity)
    float fpr = (float)fp_m / (float)(fp_m + tn); // False Positive Rate
    float fnr = (float)fn / (float)(fn + tp); // False Negative Rate
    float ppv = (float)tp / (float)(tp + fp_m); // Positive Predictive Value (Precision)
    float npv = (float)tn / (float)(tn + fn); // Negative Predictive Value
    float balanced_accuracy = (tpr + tnr) / 2;
    float f1_score = 2 * (ppv * tpr) / (ppv + tpr);


    printf("Testing completed. \n");
    printf("True Positives: %d \n", tp);
    printf("True Negatives: %d \n", tn);
    printf("False Positives: %d \n", fp_m);
    printf("False Negatives: %d \n", fn);
    printf("Accuracy: %f \n", accuracy);
    printf("True Positive Rate (Sensitivity): %f \n", tpr);
    printf("True Negative Rate (Specificity): %f \n", tnr);
    printf("False Positive Rate: %f \n", fpr);
    printf("False Negative Rate: %f \n", fnr);
    printf("Positive Predictive Value (Precision): %f \n", ppv);   
    printf("Negative Predictive Value: %f \n", npv);
    printf("Balanced Accuracy: %f \n", balanced_accuracy);
    printf("F1 Score: %f \n", f1_score);

    free(line);

    // *********************************************************************
    // ************************ FREE ALLOCATED MEMORY **********************
    // *********************************************************************

    printf("Freeing allocated memory... \n");
    for(int i = 0; i < FEATURES; i++) { 
        for(int j = 0; j < M; j++) {
            free(vector_matrices[i][j]); // Free each component of the HDV
        }
        free(vector_matrices[i]); // Free M rows, each level HDV
    }
    free(vector_matrices); // Free array of pointers to 2D matrices, to every feature matrix
    free(line);

    return 0; 
}