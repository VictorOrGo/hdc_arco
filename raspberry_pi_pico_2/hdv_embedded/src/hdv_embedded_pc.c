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
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#define M 39
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
    float lower;
    float higher;
} range_t;

int count_differences(int16_t *hdv1, int16_t *hdv2, int dim) {
    int diff = 0;
    for (int i = 0; i < dim; i++) {
        if (hdv1[i] != hdv2[i]) diff++;
    }
    return diff;
}

uint16_t range_hdv_levels(char **matrix_f, int num_features, int m_levels){
    
    gen_random_encoded_hdv(matrix_f[0]);
    int16_t hdv_array[HDV_DIM];
    decode_hdv_to_array(matrix_f[0], hdv_array);
    
    int b = HDV_DIM / (2 * (m_levels - 1));
    if (b == 0) {
        b = 1;
    }

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
            uint16_t num = get_number_from_hdv(matrix_f[i], indices_selected[k]);
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

void get_hdv_level(char **matrix_f, char *result_hdv, float lower, float higher, float value, uint16_t m_levels){

    float increment = (fabs(higher) + fabs(lower)) / m_levels; // Range divided by number of levels
    float level_lower_bound, level_upper_bound;
    if(value < lower){
        memcpy(result_hdv, matrix_f[0], LENGTH_HDV_BYTES * sizeof(char)); // Copy the corresponding HDV to result_hdv
    } else if (value > higher) {
        memcpy(result_hdv, matrix_f[m_levels-1], LENGTH_HDV_BYTES * sizeof(char)); // Copy the corresponding HDV to result_hdv
    }

    for(int i = 0; i < m_levels; i++){
        level_lower_bound = lower + (i * increment);
        level_upper_bound = lower + ((i + 1) * increment);
        
        if(value >= level_lower_bound && value < level_upper_bound){
            memcpy(result_hdv, matrix_f[i], LENGTH_HDV_BYTES * sizeof(char)); // Copy the corresponding HDV to result_hdv
        }
    }
    
}

void fill_json_data(const char *line, json_data_t *data) {
    cJSON *json = cJSON_Parse(line);
    if (!json) {
        fprintf(stderr, "Error parsing JSON line\n");
        return;
    }

    #define GET_FLOAT(name, json_field) \
        do { \
            cJSON *item = cJSON_GetObjectItemCaseSensitive(json, json_field); \
            if (item == NULL || cJSON_IsNull(item)) { \
                data->name = INT_MIN; \
            } else if (cJSON_IsNumber(item)) { \
                data->name = (float)item->valuedouble; \
            } else { \
                data->name = INT_MIN; \
            } \
        } while(0)

    #define GET_CHAR(name, json_field) \
        do { \
            cJSON *item = cJSON_GetObjectItemCaseSensitive(json, json_field); \
            if (item && cJSON_IsString(item) && item->valuestring && item->valuestring[0] != '\0') { \
                data->name = item->valuestring[0]; \
            } else { \
                data->name = '\0'; \
            } \
        } while(0)

    GET_FLOAT(travel_ms, "travel_ms");
    GET_FLOAT(hops, "hops");
    GET_FLOAT(hours_in_emergency, "hours_in_emergency");
    GET_FLOAT(hours_in_power, "hours_in_power");
    GET_FLOAT(times_in_emergency, "times_in_emergency");
    GET_FLOAT(times_in_power, "times_in_power");
    GET_FLOAT(battery_level, "battery_level");
    GET_FLOAT(link_quality, "link_quality");
    GET_FLOAT(WBN_rssi_correction_val, "WBN_rssi_correction_val");
    GET_FLOAT(cbmac_blacklisting_channels_min_to_40, "cbmac_blacklisting_channels_min_to_40");
    GET_FLOAT(cluster_channel, "cluster_channel");
    GET_FLOAT(scanstat_avg_routers, "scanstat_avg_routers");
    GET_FLOAT(network_scans_amount, "network_scans_amount");

    GET_FLOAT(trace_options_sequence, "trace_options.sequence");

    GET_FLOAT(cbmac_details_cbmac_load, "cbmac_details.cbmac_load");
    GET_FLOAT(cbmac_details_cbmac_rx_messages_ack, "cbmac_details.cbmac_rx_messages_ack");
    GET_FLOAT(cbmac_details_cbmac_rx_messages_unack, "cbmac_details.cbmac_rx_messages_unack");
    GET_FLOAT(cbmac_details_cbmac_rx_ack_other_reasons, "cbmac_details.cbmac_rx_ack_other_reasons");
    GET_FLOAT(cbmac_details_cbmac_tx_ack_cca_fail, "cbmac_details.cbmac_tx_ack_cca_fail");
    GET_FLOAT(cbmac_details_cbmac_tx_ack_not_received, "cbmac_details.cbmac_tx_ack_not_received");
    GET_FLOAT(cbmac_details_cbmac_tx_messages_ack, "cbmac_details.cbmac_tx_messages_ack");
    GET_FLOAT(cbmac_details_cbmac_tx_messages_unack, "cbmac_details.cbmac_tx_messages_unack");
    GET_FLOAT(cbmac_details_cbmac_tx_cca_unack_fail, "cbmac_details.cbmac_tx_cca_unack_fail");

    GET_FLOAT(buffer_usage_average, "buffer_usage.average");

    GET_FLOAT(nexthop_details_advertised_cost, "nexthop_details.advertised_cost");
    GET_FLOAT(nexthop_details_next_hop_quality, "nexthop_details.next_hop_quality");
    GET_FLOAT(nexthop_details_next_hop_rssi, "nexthop_details.next_hop_rssi");
    GET_FLOAT(nexthop_details_next_hop_power, "nexthop_details.next_hop_power");

    GET_FLOAT(installation_quality_quality_indicator, "Installation quality.quality_indicator");

    GET_FLOAT(nexthop_details_next_hop_address, "nexthop_details.next_hop_address");
    GET_FLOAT(cfmac_pending_broadcast_le_member, "cfmac_pending_broadcast_le_member");
    GET_FLOAT(unack_broadcast_channel, "Unack_broadcast_channel");

    GET_CHAR(classification, "classification");

    #undef GET_FLOAT
    #undef GET_CHAR

    cJSON_Delete(json);
}

float std_desv(float data[], int n, float mean) {
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += pow(data[i] - mean, 2);
    }
    return sqrt(sum / n);
}

int main() {
    srand(106); // Seed for reproducibility

    printf("D = %d M = %d\n", HDV_DIM, M);

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

    // *********************************************************************
    // ***************************** TRAINING ******************************
    // *********************************************************************

    FILE *fp = fopen("/home/victor/hdc_arco/lum_75_train.json", "r");
    if (!fp) {
        perror("Error opening file");
        return 1;
    }

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

    while (getline(&line, &len, fp) != -1) {
        fill_json_data(line, &data); // Fill the struct with the data from the JSON line
        
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
        
    }

    normalize_bipolar_hdv(prot_hdv); // Final normalization of the prototype HDV

    end = clock();
    elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Training time: %f (s)\n", elapsed);

    printf("Training completed. \n");
    printf("Number of lines processed: %d \n", num_line);
    printf("Number of lines skipped due to NaN values: %d \n", num_lines_skipped);

    fclose(fp);

    // *********************************************************************
    // ********************** THRESHOLD CALCULATION ************************
    // *********************************************************************

    fp = fopen("/home/victor/hdc_arco/lum_75_train.json", "r");
    if (!fp) {
        perror("Error opening file");
        return 1;
    }

    float cosine_similarities [num_line - num_lines_skipped];
    line = NULL;
    num_line = 0;
    num_lines_skipped = 0;
    len = 0;
    ptr = &data.travel_ms; // Pointer to the first field of the struct int *ptr = (int *)&data; 
    float threshold = 0;
    float cosine_mean = 0;
    float cosine_stddev = 0;
    int tot = 0;
    memset(sample_hdv, 0, sizeof(sample_hdv));

    while (getline(&line, &len, fp) != -1) {
        fill_json_data(line, &data); // Fill the struct with the data from the JSON line

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

        normalize_bipolar_hdv(sample_hdv); // Normalize the sample HDV

        cosine_similarities[tot] = cosine_similarity(prot_hdv, sample_hdv); 
        cosine_mean += cosine_similarities[tot];
        tot++;
        memset(sample_hdv, 0, sizeof(sample_hdv));
    }

    fclose(fp);

    cosine_mean = cosine_mean / tot;
    cosine_stddev = std_desv(cosine_similarities, tot, cosine_mean);

    threshold = cosine_mean - (2 * cosine_stddev);

    printf("Threshold calculated: %f \n", threshold);
    printf("Cosine mean: %f \n", cosine_mean);
    printf("Cosine standard deviation: %f \n", cosine_stddev);

    // *********************************************************************
    // ****************************** TESTING ******************************
    // *********************************************************************

    // Variables for machine learning metrics calculation such as accuracy, precision, recall, F1-score, etc.
    int tp = 0; // True positives
    int tn = 0; // True negatives

    fp = fopen("/home/victor/hdc_arco/lum_75_test_combined_005.json", "r");
    if (!fp) {
        perror("Error opening file");
        return 1;
    }

    float cos_similirity = 0;
    num_line = 0;
    num_lines_skipped = 0;
    len = 0;
    int total = 0;
    int real_entry = 0;
    int synthetic_entry = 0;
    char real_classification = '\0';
    memset(sample_hdv, 0, sizeof(sample_hdv));

    start = clock();
    while (getline(&line, &len, fp) != -1) {
        fill_json_data(line, &data); // Fill the struct with the data from the JSON line

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

            if (value == INT_MIN || value == '\0') { // Check for Null value in json 
                num_lines_skipped++;
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

        if (value == INT_MIN || value == '\0') { // Check for Null value in json
            printf("Skipping line %d due to NaN values. \n", num_line);
            num_lines_skipped++;
            continue;
        }

        cos_similirity = cosine_similarity(prot_hdv, sample_hdv); // Calculate cosine similarity between prototype HDV and sample HDV
        
        if (cos_similirity >= threshold && real_classification == 'r') {
            tp++;
        }
        else if (cos_similirity < threshold && real_classification == 's') {
            tn++;
        }
 
        total++;
        memset(sample_hdv, 0, sizeof(sample_hdv));
        
    }

    end = clock();
    elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Testing time: %f (s)\n", elapsed);

    size_t total_mem = FEATURES * sizeof(char**) + FEATURES * M * sizeof(char*) + FEATURES * M * LENGTH_HDV_BYTES * sizeof(char);
    printf("Memory matrix: %zu bytes (%ld KB, %ld MB)\n", total_mem, total_mem / 1024, total_mem / (1024 * 1024));

    printf("Number of test samples processed: %d \n", total);
    printf("Number of test samples skipped due to NaN values: %d \n", num_lines_skipped);
    printf("Number of real test samples: %d \n", real_entry);
    printf("Number of synthetic test samples: %d \n", synthetic_entry);
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

