#!/bin/bash
cp zip_files_csv.py /home/tvhahn/scratch/_temp_random_search_results/zip_files_csv.py
cd /home/tvhahn/scratch/_temp_random_search_results

module load scipy-stack/2019b
python zip_files_csv.py

## move all .zip files in interim_data_sample to main directory
mv *.zip /home/tvhahn/projects/def-mechefsk/tvhahn/_parameter_search_results