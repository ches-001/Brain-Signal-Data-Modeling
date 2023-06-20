# **EEG (Electroencephalography) / NIRS (Near Infrared Spectroscopy) Data Modeling**

## How to run:
1. Download the EEG and NIRS datasets from the [dataset url](https://depositonce.tu-berlin.de/items/eb49d988-33d4-4eac-9ed3-36dcd711f9d2)

2. Copy the downloaded EEG and NIRS dataset into the /data/samples folder on the same directory level as the notebook, each dataset (EEG and NIRS) should be in their respective folders named EEG and NIRS respectively.

3. Run the 'data_preparation/segment_generator.py' script to extract the zip files and generate segments and csv files for the EEG and NIRS segments, which will subsequently be used by the dataset class for retrieving data. <br>
This script takes in 6 CLI parameters, use the "-h" or "--help" flag to see what each parameter do.

4. If all is set, open the "data_modeling.ipynb" notebook, set the parameters accordinly, including the paths, and run it.

Cheers.