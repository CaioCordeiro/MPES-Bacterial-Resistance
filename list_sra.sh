#!/bin/bash

# Script to retrieve SRA IDs for a specific bacteria species using NCBI Datasets CLI

# --- Configuration ---
bacteria_species="Escherichia coli"  # Replace with your desired species
output_directory="ecoli_sra_metadata" # Directory to store downloaded metadata
metadata_zip_file="ecoli_metadata.zip"
sra_ids_file="ecoli_sra_ids.txt"
# --- End Configuration ---

# Ensure jq is installed (for parsing JSON)
if ! command -v jq &> /dev/null
then
    echo "Error: jq is not installed. Please install it (e.g., 'sudo apt-get install jq' or 'sudo yum install jq')."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$output_directory"

echo "Step 1: Searching for genome assemblies of '$bacteria_species'..."
genome_accessions=$(datasets summary genome taxon "$bacteria_species" | grep -o 'GCA_........')

if [ -z "$genome_accessions" ]; then
    echo "No genome assemblies found for '$bacteria_species'."
    exit 1
fi

genome_accessions=$(echo "$genome_accessions" | awk '{print $1}')

echo "Found the following genome accessions:"
echo "$genome_accessions"

echo "Step 2: Downloading SRA metadata for these genome assemblies..."
datasets download genome --accession "$genome_accessions" --include sra-metadata --output "$output_directory/$metadata_zip_file"

if [ ! -f "$output_directory/$metadata_zip_file" ]; then
    echo "Error: Failed to download SRA metadata."
    exit 1
fi

echo "SRA metadata downloaded to '$output_directory/$metadata_zip_file'."

echo "Step 3: Extracting and parsing SRA metadata to get SRA IDs..."
cd "$output_directory" || exit 1

# Unzip the metadata file
unzip -q "$metadata_zip_file"

# Find JSON files that might contain SRA metadata (adjust pattern if needed)
sra_json_files=$(find . -name "*sra_metadata.json" -o -name "*experiment.json")

if [ -z "$sra_json_files" ]; then
    echo "Warning: No SRA metadata JSON files found in the extracted directory."
    echo "Please check the contents of '$metadata_zip_file' to adjust the file search."
    cd .. || exit 1
    exit 1
fi

sra_ids=()

while IFS= read -r file; do
    echo "Processing metadata file: '$file'"
    if grep -q '"srr_accession":' "$file"; then
        extracted_ids=$(cat "$file" | jq -r '.Runs[].Run' | jq -r '.Accession')
        if [ ! -z "$extracted_ids" ]; then
            sra_ids+=($(echo "$extracted_ids"))
        fi
    elif grep -q '"run_accession":' "$file"; then
        extracted_ids=$(cat "$file" | jq -r '.experiments[].runs[].run_accession')
        if [ ! -z "$extracted_ids" ]; then
            sra_ids+=($(echo "$extracted_ids"))
        fi
    fi
done <<< "$sra_json_files"

cd .. || exit 1

# Remove duplicate SRA IDs
unique_sra_ids=$(printf '%s\n' "${sra_ids[@]}" | sort -u)

# Save the unique SRA IDs to a file
echo "$unique_sra_ids" > "$sra_ids_file"

echo "Step 4: Unique SRA IDs for '$bacteria_species' have been saved to '$sra_ids_file'."

# Optional: Clean up the downloaded metadata
# rm -rf "$output_directory"
# echo "Cleaned up downloaded metadata."

echo "Script finished."