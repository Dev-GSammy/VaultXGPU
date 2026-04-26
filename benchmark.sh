#!/bin/bash

# Configuration
OUTPUT_CSV="benchmark_results.csv"
BINARY="./vaultx_sycl"
PLOT_DIR="/nfs_nvme/sfatunmbi/plots"
REPEATS=3
K_START=27
K_END=32

# Ensure the plot directory exists
mkdir -p "$PLOT_DIR"

# Force CPU selection for SYCL
export ONEAPI_DEVICE_SELECTOR=opencl:cpu

# Write CSV Header
echo "K,Run,T1_Seconds,T2_Sort_Seconds,Write_Seconds,Total_Seconds" > "$OUTPUT_CSV"

echo "Starting benchmarks on EPYC CPU..."

for k in $(seq $K_START $K_END); do
    for run in $(seq 1 $REPEATS); do
        echo "------------------------------------------"
        echo "Running: K=$k | Attempt: $run/$REPEATS"
        
        # Capture the raw output
        RESULT=$($BINARY -k $k -f "$PLOT_DIR" 2>&1)
        
        # Log to terminal so you can see it's working
        echo "$RESULT" | grep "done" || echo "$RESULT" | grep "Summary" -A 4

        # Flexible parsing: looks for the line, then pulls the number
        T1=$(echo "$RESULT" | grep "Table1:" | awk '{print $2}')
        T2=$(echo "$RESULT" | grep "Sort+Table2:" | awk '{print $2}')
        WR=$(echo "$RESULT" | grep "Write:" | awk '{print $2}')
        TOT=$(echo "$RESULT" | grep "Total:" | awk '{print $2}')

        # Clean up any remaining whitespace
        T1=$(echo $T1 | xargs); T2=$(echo $T2 | xargs); WR=$(echo $WR | xargs); TOT=$(echo $TOT | xargs)

        # Fallback to 0 if empty
        T1=${T1:-0}; T2=${T2:-0}; WR=${WR:-0}; TOT=${TOT:-0}

        # Save to CSV
        echo "$k,$run,$T1,$T2,$WR,$TOT" >> "$OUTPUT_CSV"
        
        echo "Saved to CSV -> Total: ${TOT}s"
    done
done

echo "------------------------------------------"
echo "All tests complete. Results in $OUTPUT_CSV"
