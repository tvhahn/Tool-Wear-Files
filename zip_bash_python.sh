#!/bin/bash
cd interim_sample_data

python /home/tim/Documents/Tool-Wear-Files/interim_sample_data/zip_files.py

## move all .zip files in interim_data_sample to main directory
mv *.zip /home/tim/Documents/Tool-Wear-Files/zip_interim_sample_data