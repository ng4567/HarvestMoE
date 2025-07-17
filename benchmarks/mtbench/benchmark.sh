#!/bin/bash

OUTPUT_CSV="bench_results.csv"
COMMAND="python bench.py --port 30000 --max-new-tokens 32 --ubs 324 --n-ub 14"

# Write CSV header only if file doesn't exist
if [ ! -f "$OUTPUT_CSV" ]; then
    echo "time,throughput" > "$OUTPUT_CSV"
fi

for i in {1..10}
do
    # Run the command and capture output
    OUTPUT=$($COMMAND)

    # Extract time and throughput (adjust the patterns as needed)
    # Example assumes lines like: "Time: 12.34s" and "Throughput: 56.78"
    TIME=$(echo "$OUTPUT" | grep -i "time" | head -1 | awk '{print $2}' | tr -d 's')
    THROUGHPUT=$(echo "$OUTPUT" | grep -i "throughput" | head -1 | awk '{print $2}')

    # Append to CSV
    echo "$TIME,$THROUGHPUT" >> "$OUTPUT_CSV"
done
