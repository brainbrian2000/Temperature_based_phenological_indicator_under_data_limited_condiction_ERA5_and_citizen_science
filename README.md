# Modeling_Spider_Tree
Y.S. Huang, C.H. Fan, Y.C. Lu, Y.T. Wu, Y.C. Niu, Y.H. Hsieh, Y.K. Liao*, S.J. Sun*. Climate-driven phenology modeling for data-limited species using ERA5 reanalysis and citizen science. (in prep)
## Maintainer
name: C.H. Fan
## environment
Ubuntu-24.04
Python3.11

## How to use
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
