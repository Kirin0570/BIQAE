#!/bin/bash

# Loop through all directories matching the pattern processing_BL_*p*
for dir in processing_BL_*p*; do
    # Check if the run_algo directory exists inside the current directory
    if [ -d "$dir/run_algo" ]; then
        # Navigate to the run_algo directory and run condor_submit
        (cd "$dir/run_algo" && condor_submit submit_run_algo.sub)
        echo "Executed condor_submit in $dir/run_algo"
    fi
done
