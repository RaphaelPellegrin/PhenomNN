#!/bin/bash

# Local bash script to run experiments with different encodings.

# Create log file
LOG_FILE="log_experiments.txt"

# Clear the log file
> $LOG_FILE

# Define datasets
DATASETS=(
    "coauthorship_with_encodings/cora"
    "coauthorship_with_encodings/dblp"
    "cocitation_with_encodings/citeseer"
    "cocitation_with_encodings/cora"
    "cocitation_with_encodings/pubmed"
)

# Define encoding types (the part after features_<dataclass>_<dataset>_)
ENCODING_TYPES=(
    "degree_encodings_normalized_True"
    "laplacian_encodings_Normalized_normalized_True"
    "laplacian_encodings_Hodge_normalized_True"
    "rw_encodings_EN_k_20_normalized_True"
    "rw_encodings_WE_k_20_normalized_True"
    "rw_encodings_EE_k_20_normalized_True"
    "curvature_encodings_FRC_normalized_True"
    "curvature_encodings_ORC_normalized_True"
)

# Function to log messages
log_message() {
    echo "$1" | tee -a $LOG_FILE
}

# Function to get dataclass and dataset name from path
get_dataclass_dataset() {
    local dataset_path=$1
    local dataclass=$(echo $dataset_path | cut -d'/' -f1 | sed 's/_with_encodings//')
    local dataset_name=$(echo $dataset_path | cut -d'/' -f2)
    echo "$dataclass $dataset_name"
}

# Function to construct encoding name
construct_encoding_name() {
    local dataclass=$1
    local dataset_name=$2
    local encoding_type=$3
    echo "features_${dataclass}_${dataset_name}_${encoding_type}"
}

# Function to check if encoding exists for dataset
encoding_exists() {
    local dataset=$1
    local encoding=$2
    local encoding_file="data/$dataset/${encoding}.pkl"
    [ -f "$encoding_file" ]
}

# Log start time
log_message "=== Starting experiments at $(date) ==="
log_message ""

# Loop through each dataset
for dataset in "${DATASETS[@]}"; do
    log_message "=========================================="
    log_message "Processing dataset: $dataset"
    log_message "=========================================="
    log_message ""
    
    # Get dataclass and dataset name
    read dataclass dataset_name <<< $(get_dataclass_dataset "$dataset")
    log_message "Dataclass: $dataclass, Dataset: $dataset_name"
    log_message ""
    
    # First run with original features
    log_message "------------------------------------------"
    log_message "Running experiment for dataset: $dataset"
    log_message "Using encoding: features (original)"
    log_message "------------------------------------------"
    log_message ""
    
    COMMAND="python train_faster.py --type phenomnn_s --activate_dataset $dataset --encoding features --lr 0.01 --dropout 0.7 --hidden 64 --lam0 20 --lam1 80 --alp 0.1 --prop_step 16 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu -1 --sigma -1"
    log_message "Command: $COMMAND"
    log_message ""
    
    log_message "=== Output for $dataset with features (original) ==="
    eval $COMMAND 2>&1 | tee -a $LOG_FILE
    
    log_message ""
    log_message "=== Finished $dataset with features (original) at $(date) ==="
    log_message ""
    log_message ""
    
    # Loop through each encoding type
    for encoding_type in "${ENCODING_TYPES[@]}"; do
        # Construct the full encoding name
        encoding=$(construct_encoding_name "$dataclass" "$dataset_name" "$encoding_type")
        
        # Check if this encoding exists for this dataset
        if encoding_exists "$dataset" "$encoding"; then
            log_message "------------------------------------------"
            log_message "Running experiment for dataset: $dataset"
            log_message "Using encoding: $encoding"
            log_message "------------------------------------------"
            log_message ""
            
            # Log the command being executed
            COMMAND="python train_faster.py --type phenomnn_s --activate_dataset $dataset --encoding $encoding --lr 0.01 --dropout 0.7 --hidden 64 --lam0 20 --lam1 80 --alp 0.1 --prop_step 16 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu -1 --sigma -1"
            log_message "Command: $COMMAND"
            log_message ""
            
            # Run the command and capture all output
            log_message "=== Output for $dataset with $encoding ==="
            eval $COMMAND 2>&1 | tee -a $LOG_FILE
            
            log_message ""
            log_message "=== Finished $dataset with $encoding at $(date) ==="
            log_message ""
            log_message ""
        else
            log_message "Skipping encoding $encoding for dataset $dataset (file not found)"
            log_message ""
        fi
    done
    
    log_message "=== Completed all encodings for $dataset ==="
    log_message ""
done

log_message "=== All experiments completed at $(date) ===" 