#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cjson/cJSON.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <malloc.h>
#include "hdc.h"
#include "bytehd.h"
#include <limits.h>

#define M 3336
#define NORMALIZATION_SUM 100
#define THRESHOLD 0.90
#define FEATURES 17

typedef struct {
    int battery_level;
    int hops;
    int hours_in_emergency;
    int hours_in_power;
    int link_quality;
    int outputState; 
    int state;
    int times_in_emergency;
    int times_in_power;
    int travel_ms;
    int two_in_one_battery_level;
    int WBN_rssi_correction_val;
    int network_scans_amount; 
    int cfmac_pending_broadcast_le_member;
    int cluster_channel;
    int Unack_broadcast_channel;
    int cluster_members;
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
                rand_indice = rand() % HDV_DIM;
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

void fill_json_data(const char *line, json_data_t *data) {
    cJSON *json = cJSON_Parse(line);
    if (!json) {
        fprintf(stderr, "Error parsing JSON line\n");
        return;
    }

    // For each field we obtain the value if it exists, or NaN if it is null or it doesn't exist
    #define GET_INT(name, json_field) \
        do { \
            cJSON *item = cJSON_GetObjectItemCaseSensitive(json, json_field); \
            if (item == NULL || cJSON_IsNull(item)) { \
                data->name = INT_MIN; \
            } else if (cJSON_IsNumber(item)) { \
                data->name = item->valuedouble; \
            } else { \
                data->name = INT_MIN; \
            } \
        } while(0)

    GET_INT(battery_level, "battery_level");
    GET_INT(hops, "hops");
    GET_INT(hours_in_emergency, "hours_in_emergency");
    GET_INT(hours_in_power, "hours_in_power");
    GET_INT(link_quality, "link_quality");
    GET_INT(outputState, "outputState");
    GET_INT(state, "state");
    GET_INT(times_in_emergency, "times_in_emergency");
    GET_INT(times_in_power, "times_in_power");
    GET_INT(travel_ms, "travel_ms");
    GET_INT(two_in_one_battery_level, "two_in_one_battery_level");
    GET_INT(WBN_rssi_correction_val, "WBN_rssi_correction_val");
    //GET_INT(buffer_usage_average, "buffer_usage.average");
    //GET_INT(buffer_usage_maximum, "buffer_usage.maximum");
    //GET_INT(Network_channel_PER, "Network_channel_PER");
    //GET_INT(cbmac_load, "cbmac_details.cbmac_load");
    //GET_INT(cbmac_rx_messages_ack, "cbmac_details.cbmac_rx_messages_ack");
    //GET_INT(cbmac_rx_messages_unack, "cbmac_details.cbmac_rx_messages_unack");
    //GET_INT(cbmac_rx_ack_other_reasons, "cbmac_details.cbmac_rx_ack_other_reasons");
    //GET_INT(cbmac_tx_ack_cca_fail, "cbmac_details.cbmac_tx_ack_cca_fail");
    //GET_INT(cbmac_tx_ack_not_received, "cbmac_details.cbmac_tx_ack_not_received");
    //GET_INT(cbmac_tx_messages_ack, "cbmac_details.cbmac_tx_messages_ack");
    //GET_INT(cbmac_tx_messages_unack, "cbmac_details.cbmac_tx_messages_unack");
    //GET_INT(cbmac_tx_cca_unack_fail, "cbmac_details.cbmac_tx_cca_unack_fail");
    //GET_INT(unknown, "unknown");
    //GET_INT(unkown, "unkown");
    GET_INT(network_scans_amount, "network_scans_amount");
    //GET_INT(scanstat_avg_routers, "scanstat_avg_routers");
    GET_INT(cfmac_pending_broadcast_le_member, "cfmac_pending_broadcast_le_member");
    GET_INT(cluster_channel, "cluster_channel");
    //GET_INT(packets_dropped, "packets_dropped");
    GET_INT(Unack_broadcast_channel, "Unack_broadcast_channel");
    GET_INT(cluster_members, "cluster_members");
    //GET_INT(nexthop_advertised_cost, "nexthop_details.advertised_cost");
    //GET_INT(nexthop_sink_address, "nexthop_details.sink_address");
    //GET_INT(nexthop_next_hop_address, "nexthop_details.next_hop_address");
    //GET_INT(nexthop_next_hop_quality, "nexthop_details.next_hop_quality");
    //GET_INT(nexthop_next_hop_rssi, "nexthop_details.next_hop_rssi");
    //GET_INT(nexthop_next_hop_power, "nexthop_details.next_hop_power");
    //GET_INT(src, "src");

    #undef GET_INT

    cJSON_Delete(json);
}

int main() {

    // *********************************************************************
    // ***************** SEND DATA THROUGH USB TO THE PICO *****************
    // *********************************************************************

    range_t ranges[] = {
        {"battery_level", 2810, 4102}, //si
        {"hops", -9, 11},  //si
        {"hours_in_emergency", -10, 10}, //si
        {"hours_in_power", 3504, 4053}, //si
        {"link_quality", 112, 265}, //si
        {"outputState", 56, 16651}, //si
        {"state", -9, 11}, //si
        {"times_in_emergency", -10, 10}, //si
        {"times_in_power", -10, 10}, //si
        {"travel_ms", -10, 166}, //si
        {"two_in_one_battery_level", -10, 10}, //si
        {"WBN_rssi_correction_val", -14, -1}, //si
        {"network_scans_amount", 98, 153}, //si
        {"cfmac_pending_broadcast_le_member", 21, 41}, //si
        {"cluster_channel", -7, 46}, //si
        {"Unack_broadcast_channel", 8, 30}, //si
        {"cluster_members", -5, 15}, //si
    };


    //stdio_init_all();

    //sleep_ms(1000); // Wait for USB to be ready

    // *********************************************************************
    // ********************* ALLOCATE MEMORY FOR MATRICES ******************
    // *********************************************************************

    char ***vector_matrices; 
    // vector_matrices[i] points to the matrix i (feature i)
    // vector_matrices[i][j] points to the hypervector j (level hdv) of the matrix i (feature i)
    // vector_matrices[i][j][k] points to the component k (component k) of the hypervector j (level hdv) of the matrix i (feature i)
    
    //char vector_matrices[FEATURES][M][LENGTH_HDV_BYTES]; 

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

    // *********************************************************************
    // ***************************** TRAINING ******************************
    // *********************************************************************

    FILE *fp = fopen("/home/victor/hdc_arco/lum15Train.json", "r");
    if (!fp) {
        perror("Error opening file");
        return 1;
    }

    int8_t prot_hdv[HDV_DIM];
    int8_t sample_hdv[HDV_DIM];
    char aux_hdv[LENGTH_HDV_BYTES];
    int8_t aux_hdv_decoded[HDV_DIM];
    char *line = NULL;
    int num_line = 0;
    int num_lines_skipped = 0;
    size_t len = 0;
    json_data_t data;
    int lower;
    int higher;
    int *ptr = &data.battery_level; // Pointer to the first field of the struct int *ptr = (int *)&data; 
    int value;
    while (getline(&line, &len, fp) != -1) {
        fill_json_data(line, &data); // Fill the struct with the data from the JSON line

        for (int i = 0; i < FEATURES; i++) { // Creation of the sample HDV by bundling the HDVs of each feature
            value =  ptr[i]; // get the value of the field

            if (value == INT_MIN) { // Check for Null value in json
                num_lines_skipped++;
                break;
            }
    
            lower = ranges[i].lower;
            higher = ranges[i].higher;
            if (get_hdv_level(vector_matrices[i], aux_hdv, lower, higher, value, M) != 0) {
                printf("Error: get_hdv_level returned error value for feature %d with value %d.\n", i, value);
                return 1;
            }

            decode_hdv_to_array(aux_hdv, aux_hdv_decoded); // Decode the HDV from bytes to int8_t array
            
            if (i == 0) {
                memcpy(sample_hdv, aux_hdv_decoded, HDV_DIM * sizeof(int8_t)); // Initialize sample_hdv with the first feature HDV
                continue;
            }
            bundle_hdv(sample_hdv, aux_hdv_decoded, sample_hdv); // Accumulate the HDVs of each feature
        }

        normalize_bipolar_hdv(sample_hdv); // Normalize the sample HDV

        if(num_line == 0) {
            memcpy(prot_hdv, sample_hdv, HDV_DIM * sizeof(int8_t)); // Initialize prototype HDV with the first sample HDV
        }
        else if (num_line >= NORMALIZATION_SUM) {
            bundle_hdv(prot_hdv, sample_hdv, prot_hdv); // Accumulate the sample HDVs to create the prototype HDV
            normalize_bipolar_hdv(prot_hdv); // Normalize the prototype HDV
        }
        else {
            bundle_hdv(prot_hdv, prot_hdv, sample_hdv); // Accumulate the sample HDVs to create the prototype HDV
        }
        num_line++;
        
    }

    printf("Training completed. \n");
    printf("Number of lines processed: %d \n", num_line);
    printf("Number of lines skipped due to NaN values: %d \n", num_lines_skipped);

    fclose(fp);

    // *********************************************************************
    // ****************************** TESTING ******************************
    // *********************************************************************

    fp = fopen("/home/victor/hdc_arco/lum15Test.json", "r");
    if (!fp) {
        perror("Error opening file");
        return 1;
    }

    float cos_similirity = 0;
    int samples_over_threshold = 0;
    num_line = 0;
    num_lines_skipped = 0;
    len = 0;
    while (getline(&line, &len, fp) != -1) {
        fill_json_data(line, &data); // Fill the struct with the data from the JSON line

        for (int i = 0; i < FEATURES; i++) { // Creation of the sample HDV by bundling the HDVs of each feature
            value =  ptr[i]; // get the value of the field

            if (value == INT_MIN) { // Check for Null value in json
                num_lines_skipped++;
                break;
            }
    
            lower = ranges[i].lower;
            higher = ranges[i].higher;
            if (get_hdv_level(vector_matrices[i], aux_hdv, lower, higher, value, M) != 0) {
                printf("Error: get_hdv_level returned error value for feature %d with value %d.\n", i, value);
                return 1;
            }

            decode_hdv_to_array(aux_hdv, aux_hdv_decoded); // Decode the HDV from bytes to int8_t array
            
            if (i == 0) {
                memcpy(sample_hdv, aux_hdv_decoded, HDV_DIM * sizeof(int8_t)); // Initialize sample_hdv with the first feature HDV
                continue;
            }
            bundle_hdv(sample_hdv, aux_hdv_decoded, sample_hdv); // Accumulate the HDVs of each feature
        }

        num_line++;

        if (value == INT_MIN) { // Check for Null value in json
            continue;
        }

        normalize_bipolar_hdv(sample_hdv); // Normalize the sample HDV

        cos_similirity = cosine_similarity(prot_hdv, sample_hdv); // Calculate cosine similarity between prototype HDV and sample HDV
        printf("Cosine similarity of sample %d with prototype HDV: %f \n", num_line, cos_similirity);
        if (cos_similirity >= THRESHOLD) {
            samples_over_threshold++;
        }
        
    }

    printf("Testing completed. \n");
    printf("Percentage of samples over threshold: %2f %% \n", (samples_over_threshold / (num_line - num_lines_skipped)) * 100.0);
    printf("Number of lines processed: %d \n", num_line);
    printf("Number of lines skipped due to NaN values: %d \n", num_lines_skipped);

    free(line);
    fclose(fp);

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
    
    return 0;
}

/*print data struct
        printf("Data: battery_level: %d, hops: %d, hours_in_emergency: %d, hours_in_power: %d, link_quality: %d, outputState: %d, state: %d, times_in_emergency: %d, times_in_power: %d, travel_ms: %d, two_in_one_battery_level: %d, WBN_rssi_correction_val: %d, network_scans_amount: %d, cfmac_pending_broadcast_le_member: %d, cluster_channel: %d, Unack_broadcast_channel: %d, cluster_members: %d \n", 
            data.battery_level, data.hops, data.hours_in_emergency, data.hours_in_power, data.link_quality, data.outputState, data.state, data.times_in_emergency, data.times_in_power, data.travel_ms, data.two_in_one_battery_level, data.WBN_rssi_correction_val, data.network_scans_amount, data.cfmac_pending_broadcast_le_member, data.cluster_channel, data.Unack_broadcast_channel, data.cluster_members);
*/
