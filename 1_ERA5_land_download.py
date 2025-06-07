import cdsapi
dataset = "reanalysis-era5-land"
# request = {
#     "variable": [
#         "2m_temperature",
#         # "surface_net_solar_radiation"
#     ],
#     "year": [
#         # "2014", "2015", "2016",
#         # "2017", "2018", "2019",
#         # "2020", "2021", "2022",
#         # "2023", "2024", "2025"
#         "2014"
#     ],
#     "month": [
#         "01",
#         # "01", "02", "03",
#         # "04", "05", "06",
#         # "07", "08", "09",
#         # "10", "11", "12"
#     ],
#     "day": [
#         "01", "02", "03",
#         "04", "05", "06",
#         "07", "08", "09",
#         "10", "11", "12",
#         "13", "14", "15",
#         "16", "17", "18",
#         "19", "20", "21",
#         "22", "23", "24",
#         "25", "26", "27",
#         "28"
#     ],
#     "time": [
#         "00:00", "01:00", "02:00",
#         "03:00", "04:00", "05:00",
#         "06:00", "07:00", "08:00",
#         "09:00", "10:00", "11:00",
#         "12:00", "13:00", "14:00",
#         "15:00", "16:00", "17:00",
#         "18:00", "19:00", "20:00",
#         "21:00", "22:00", "23:00"
#     ],
#     "data_format": "netcdf",
#     "download_format": "unarchived"
# }

# client = cdsapi.Client()
# client.retrieve(dataset, request).download()

# 


import cdsapi
import os

download_dir = "./era5_land_temperature"

os.makedirs(download_dir, exist_ok=True)

client = cdsapi.Client()

years = [str(y) for y in range(2010, 2025)]
# years = [str(y) for y in range(2015, 2019)]
inverse_years = [str(y) for y in range(2024, 2013, -1)]
# years = inverse_years

months = [f"{m:02d}" for m in range(1, 13)]
# inverse_months = [f"{m:02d}" for m in range(12, 0, -1)]
# months = inverse_months

for year in years:
    for month in months:
        filename = f"era5_land_t2m_{year}_{month}.nc"
        filepath = os.path.join(download_dir, filename)

        if os.path.exists(filepath):
            print(f"exist:{filename},skip")
            continue
        filepath_zip = os.path.join(download_dir, filename[:-2] + "zip")
        if os.path.exists(filepath_zip):
            print(f"exist{filename[:-2]}zip,skip")
            continue
        print(f"downloading:{filename} ...")

        try:
            client.retrieve(
                "reanalysis-era5-land",
                {
                    "variable": "2m_temperature",
                    "year": year,
                    "month": month,
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": [f"{h:02d}:00" for h in range(24)],
                    # "area": [26, 119, 21, 123],  # 北、西、南、東（台灣）
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
                filepath,
            )
            print(f"finish:{filename}")

        except Exception as e:
            print(f"error{filename}\nreason:{e}")
