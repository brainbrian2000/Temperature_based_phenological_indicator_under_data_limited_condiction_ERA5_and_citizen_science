# Deriving exploratory temperature-based phenological indicators under data-limited conditions: integrating ERA5 and citizen science 
Y.S. Huang, C.H. Fan, Y.C. Lu, Y.T. Wu, Y.C. Niu, Y.H. Hsieh, Y.K. Liao*, S.J. Sun*. Climate-driven phenology modeling for data-limited species using ERA5 reanalysis and citizen science. (in prep)
## Maintainer
name: C.H. Fan
## environment
Ubuntu-24.04
Python3.11

## How to use
### 0. Files and system setup
#### For Run Your own datasets
- Files you need:
  - `1_ERA5_land_download.py`
  - `2_Meteorology_MinMax_avg.py`
  - `3_Processing_modeling_main.py`
  - `sample_location.csv`
    - about sample_location.csv will mention in step 2.
### 1. Download ERA5 land data
```
ERA5 land data are very large, about 2GiB per file(per month) for nc file.
Make sure disk space is enough for storage.
```

- following this guideline 
  - [User Guideline](https://cds.climate.copernicus.eu/how-to-api)
- login, and set api key in `$HOME/.cdsapirc`
- `pip install "cdsapi>=0.7.4"`
- run file 1
  - `python3 1_ERA5_land_download.py`  
- *this step maybe cost about 1 week to download it, and cost lot of storage space.*
  - maybe can create 2 or three account to download,it will more faster than only one api key in the queue.
### 2. Data Format
- ensure datasets include "Number","Date","Type","State","Coordinate"
the format shows in `sample_locations.csv`,using space between lon and lat 
### 3. Run to get 365-day temperatures of the sample points
- Run file 2
  - `python3 2_Meteorology_MinMax_avg.py`
### 4. Get final result
- Run file 3
  - `python3 3_Processing_modeling_main.py`