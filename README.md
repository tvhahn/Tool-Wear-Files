# Tool-Wear-Files
Python scripts for processing data from CNC machines.

Description of some of the files in the directory:

- **create_high_level.py** Script that is used to create a labelled high_level csv. The csv includes information on each cut (where each cut is the creation of onebrand new part). The csv can then be labelled based off of tool failure records.
- **create_split_data.py** Script that is used to split raw cut signals into individual cuts. Cuts are split by tool number, and then by cut signal (whether the cut signal is 0 or 1).
- **create_low_level.py** Script that is used to create a labelled low_level dataframe and CSV. The csv will include information of each individual split cut, along with the label (e.g. if it is failed or not).
- **1.0-data-pipeline-example.ipynb** Jupyter notebook that illustrates how the labelled low-level data table is generated.
- **high_level_LABELLED.csv** A csv that list each cut in the raw data set. It has a manually annotated label for each cut (this is just a simple example -- the annotations don't actually mean anything in this case).

## How to Process Data?
1. Make sure all the raw data files are in the raw data folder.
2. Run the 'create_high_level.py' python script. This will go through each of the raw data files, and record information about it, and save it as a csv file. That csv file can then be annotated with failure data (labels).
3. Run the 'create_split_data.py' python script. This extracts all the individual cuts from each raw data file (seperated by tool number and whether or not the tool is cutting metal). These split data files are stored as 'pickles' in a interim data folder.
4. Run the 'create_low_level.py' python script. This creates the final table of all the split cuts, with their labels.


