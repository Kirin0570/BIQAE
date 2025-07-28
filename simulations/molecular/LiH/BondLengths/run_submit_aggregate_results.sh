#!/bin/bash

# Loop through all directories matching the pattern processing_BL_*p*
for dir in processing_BL_*p*; do
    # Check if the aggregate_results directory exists inside the current directory
    if [ -d "$dir/aggregate_results" ]; then
        # Navigate to the aggregate_results directory and run condor_submit
        (cd "$dir/aggregate_results" && condor_submit submit_aggregate_results.sub)
        echo "Executed condor_submit in $dir/aggregate_results"
    fi
done
