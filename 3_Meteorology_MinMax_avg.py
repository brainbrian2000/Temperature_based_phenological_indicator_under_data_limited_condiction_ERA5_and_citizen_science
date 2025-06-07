import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4
import os
from mpl_toolkits.basemap import Basemap
import netCDF4 as nc
import matplotlib.cm as cm 
# read csv get spider tree flowering id, date and lat long
# 1,3/10/2025,Myanmar,Observation,Flowering,78," 21.98899, 96.11989",https://www.inaturalist.org/observations/264894378
# 2,3/8/2025,India,Observation,Flowering,50,"20.46345, 85.88421",https://www.inaturalist.org/observations/264783011
# 3,,,,,,,
# 4,2/21/2025,India,Observation,Flowering,797,"12.08897, 75.90360",https://www.inaturalist.org/observations/262491886
# 5,2/11/2025,India,Observation,Flowering,325," 11.10475, 77.34965",https://www.inaturalist.org/observations/261426932
# 6,1/4/2025,India,Observation,Flowering,149," 11.24756, 78.89585",https://www.inaturalist.org/observations/257487336


def read_csv(file_path):
    """
    Reads a CSV file and returns a DataFrame with the relevant columns.
    """
    df = pd.read_csv(file_path)
    df.columns = ['id', 'date', 'country', 'type', 'phenophase', 'altitude', 'lat_long', 'url']
    lat_lon_split = df['lat_long'].str.split(',', expand=True)
    df['lat'] = lat_lon_split[0].astype(float)
    df['lon'] = lat_lon_split[1].astype(float)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # 遇到無效日期會變 NaT

    df.drop(columns=['lat_long'], inplace=True)
    df.dropna(subset=['lat', 'lon', 'date'], inplace=True)  # 去除無效的 lat, lon, date
    return df
print ("Reading CSV file...")
df = read_csv('spider_tree_flowering.csv')
print ("CSV file read successfully.")
# print(df)
# print(df.loc[0]["lat_long"][0])
def drop_non_flowering(df):
    """
    Drops non-flowering observations from the DataFrame.
    """
    df = df[df['phenophase'] == 'Flowering']
    return df
print ("Dropping non-flowering observations...")
df = drop_non_flowering(df)
print ("Non-flowering observations dropped.")
# print(df)

#let negative lon turn into 360 - lon
def convert_lon(lon):
    """
    Converts longitude to a positive value.
    """
    if lon < 0:
        lon = 360 + lon
    return lon
# print ("Converting longitude...")
df['lon'] = df['lon'].apply(convert_lon)
# print ("Longitude converted.")
# print(df['lon'])
#remove url, contury
df.drop(columns=['url', 'country',"type",'phenophase'], inplace=True)

#storage into csv
df.to_csv('spider_tree_flowering_converted.csv', index=False)
#to list
df_list = df.values.tolist()
# print(df_list)
# print(int(89.2),int(-89.2))
power = 1
def get_weighting(lon,lat,lon_lat_list, power=1):

    """
    Compute normalized inverse-distance weights based only on spatial coordinates.

    Parameters:
        lon, lat         : Target coordinate
        lon_lat_list     : List of (lat, lon) known coordinates
        power            : Power for inverse distance weighting (default=1)

    Returns:
        List of weights (sum to 1), one for each input point
    """
    numerator = 0.0
    denominator = 0.0
    distance_list = []
    for i in range(len(lon_lat_list)):
        # dist = np.sqrt((lon-lon_lat_list[i][1])**2+(lat-lon_lat_list[i][0])**2)
        dist = np.hypot(lon-lon_lat_list[i][1],lat-lon_lat_list[i][0])
        if dist == 0:
            rtlist = [0,0,0,0]
            rtlist[i] = 1
            return rtlist
        distance_list.append(dist)
    power_distance_list = np.array(distance_list)**power
    inv_w = 1.0 / power_distance_list
    total_inv_w = np.sum(inv_w)
    weight = inv_w / total_inv_w
    return weight
    # total_distance = 0
    # for i in range(len(lon_lat_list)):
    #     total_distance += np.sqrt((lon-lon_lat_list[i][1])**2+(lat-lon_lat_list[i][0])**2)
    # weighting_list = []
    # for i in range(len(lon_lat_list)):
    #     weighting_list.append((np.sqrt((lon-lon_lat_list[i][1])**2+(lat-lon_lat_list[i][0])**2)/total_distance))
    # weighting_list = (1-np.array(weighting_list))/3
    # return weighting_list
def set_nearest_4point_de1(dataframe_list):
    """
    Sets the nearest 4 points for each observation in the DataFrame.
    """
    # Define the size of the grid
    grid_size = 0.1
    new_list = []
    for i in range(len(dataframe_list)):
        # Get the latitude and longitude of the observation
        lat = dataframe_list[i][3]
        lon = dataframe_list[i][4]
        # [lat1,lat2,lon1,lon2]
        lat1 = int(lat*10)/10.0
        if(lat>0):
            lat2 = int(lat*10+1)/10.0
        else:
            lat2 = int(lat*10-1)/10.0
        lon1 = int(lon*10)/10.0
        lon2 = int(lon*10+1)/10.0
        lat_lon_pair_list = []
        lat_lon_pair_list.append([lat1, lon1])
        lat_lon_pair_list.append([lat1, lon2])
        lat_lon_pair_list.append([lat2, lon1])
        lat_lon_pair_list.append([lat2, lon2])
        weighting_list = get_weighting(lon, lat, lat_lon_pair_list,power)
        month = dataframe_list[i][1].month
        year = dataframe_list[i][1].year
        date = dataframe_list[i][1].day



        new_list.append([dataframe_list[i][0],dataframe_list[i][1],year,month,date,lat,lon, lat_lon_pair_list, weighting_list])
    return new_list

df_list = set_nearest_4point_de1(df_list)
# print(df_list)
#save to csv
df_list_df = pd.DataFrame(df_list)
df_list_df.columns = ['id', 'date', 'year', 'month', 'day', 'lat', 'lon', 'lat_lon_pair_list', 'weighting_list']
df_list_df.to_csv('spider_tree_flowering_converted_4point.csv', index=False)
#ploting basemap
# plt.figure(figsize=(10, 7))

# # 設定 Basemap
# m = Basemap(projection='merc', llcrnrlat=-65, urcrnrlat=65, llcrnrlon=-180, urcrnrlon=180, lat_0 = 0, lon_0 = 180,resolution='c')

# file_name = "era5_land_t2m_"
# data = nc.Dataset("./era5_land_temperature/"+file_name+"2023"+"_"+"11"+".nc")
# print(data)
# m.drawcoastlines()
# m.drawcountries()
# # draw points by lat lon in list
# def plot_points_on_map(lat_lon_list):
#     for point in lat_lon_list:
#         print(point)
#         lat = point[4]
#         lon = point[5]
#         print(lat, lon)
#         #convert lon from 360 into +-180
#         if lon > 180:
#             lon = lon - 360
#         # plot point
#         x, y = m(lon, lat)
#         m.plot(x, y, 'bo', markersize=1)
# plt.title('Flowering Observations')
# plot_points_on_map(df_list)
# plt.savefig('flowering_observations_map.png', dpi=300)
# plt.close()


# nc_date_test = "2023-11-01"
# tree_date_test = pd.to_datetime("2023-10-1")
day_limit = 365
def is_in_days(nc_date, tree_date):
    """
    Check if the nc_date is within 180 days of the tree_date.
    """
    # Convert nc_date to datetime
    nc_date = pd.to_datetime(nc_date)
    # Calculate the difference in days
    diff = (tree_date-nc_date).days
    # print(f"Difference in days: {diff}")
    return diff <= day_limit and diff >= 0

def read_nc_year_month(year,month):
    nc_file_name = f"./era5_land_temperature/era5_land_t2m_{year}_{month:02d}.nc"
    print(f"Reading file: {nc_file_name}")
    if not os.path.exists(nc_file_name):
        print(f"File {nc_file_name} does not exist.")
        return None
    data = nc.Dataset(nc_file_name)
    return data
def read_nc_from_list(nc_data,lonlist,latlist, data_list_data):
    #data_list:[id,date,year,month,date,lat,lon, lat_lon_pair_list, weighting_list]
    #get the data form lat_lon_pair_list and weighting by weighting_list
    t2m_weighted = 0
    for i in range(len(data_list_data[7])):
        lat = data_list_data[7][i][0]
        lon = data_list_data[7][i][1]
        lat_index = np.abs(lat-latlist).argmin()
        lon_index = np.abs(lon-lonlist).argmin()
        t2m = nc_data[lat_index,lon_index]
        weighting = data_list_data[8][i]
        t2m_weighted += t2m * weighting
    #print(f"id : {data_list_data[0]} t2m_weighted: {t2m_weighted}")
    #process with nan value of temperature
    if(not t2m_weighted>0):
        t2m_weighted = 0
        t2mlist = []
        nan_list = []
        #redo process by using valid data
        lat = data_list_data[5]
        lon = data_list_data[6]
        for i in range(len(data_list_data[7])):
            lat_index = np.abs(data_list_data[7][i][0]-latlist).argmin()
            lon_index = np.abs(data_list_data[7][i][1]-lonlist).argmin()
            t2mlist.append(nc_data[lat_index,lon_index])
            #find nan
        all_nan = True
        nan_list=np.isnan(t2mlist) 
        #print(f"temp data {data_list_data[0]} t2m is nan, redo weighting")
        # print(t2mlist)
        # print(nan_list)
        for i in range(np.shape(nan_list)[0]):
            if(not nan_list[i]):
                all_nan = False
                break
        if all_nan:
            # print(f"all nan, id: {data_list_data[0]}")
            return np.nan
        
        else:
            #redo weighting
            '''
            total_distance = 0
            redo_weighting_list =[]
            points = 0
            t2m_weighted = 0
            for i in range(len(nan_list)):
                if nan_list[i]:
                    total_distance += 0
                else:
                    total_distance += np.sqrt((lon-data_list_data[7][i][1])**2+(lat-data_list_data[7][i][0])**2)
                    points += 1
            for i in range(len(nan_list)):
                if nan_list[i]:
                    redo_weighting_list.append(1)
                else:
                    redo_weighting_list.append((np.sqrt((lon-data_list_data[7][i][1])**2+(lat-data_list_data[7][i][0])**2)/total_distance))
            # calculate redo t2m_weighted
            # print(f"redo weighting list: {redo_weighting_list}")
            if(points>1):
                redo_weighting_list = (1-np.array(redo_weighting_list))/(points-1)
            for i in range(len(t2mlist)):
                if(t2mlist[i] != np.nan):
                    # print(f"t2mlist[i]: {t2mlist[i]}, redo_weighting_list[i]: {redo_weighting_list[i]}, add: {t2mlist[i] * redo_weighting_list[i]}")
                    t2m_weighted += t2mlist[i] * redo_weighting_list[i]
                else:
                    continue

            # print(f"redo weighting list: {redo_weighting_list},t2mlist: {t2mlist}, t2m_weighted: {t2m_weighted} ,points: {points}")
            # print(f"t2m: {t2m}")
            # print(f"t2mlist: {t2mlist}")
            # print(f"t2m_weighted: {t2m_weighted}")
            return t2m_weighted
            '''
            total_distance = 0
            distance_list = []
            redo_weighting_list =[]
            points = 0
            t2m_weighted = 0
            for i in range(len(nan_list)):
                if nan_list[i]:
                    total_distance += 0
                else:
                    distance_list.append(np.hypot(lon-data_list_data[7][i][1],lat-data_list_data[7][i][0]))
                    points += 1
            power_distance_list = np.array(distance_list)**power
            inv_w = 1.0 / power_distance_list
            total_inv_w = np.sum(inv_w)
            redo_weighting_list = inv_w / total_inv_w
            k = 0     #index of redo_weighting_list 
            for i in range(len(t2mlist)):
                if(t2mlist[i] != np.nan):
                    # print(f"t2mlist[i]: {t2mlist[i]}, redo_weighting_list[i]: {redo_weighting_list[i]}, add: {t2mlist[i] * redo_weighting_list[i]}")
                    t2m_weighted += t2mlist[i] * redo_weighting_list[k]
                    k += 1
                else:
                    continue

            return t2m_weighted


    else:
        return t2m_weighted
    
    

month_day_list = [31,28,31,30,31,30,31,31,30,31,30,31]
month_day_list_4 = [31,29,31,30,31,30,31,31,30,31,30,31]
temperature_previous_180days = np.zeros([len(df_list),day_limit+1])
temperature_previous_180days_min_max = np.zeros([len(df_list),day_limit+1])
temperature_previous_index = np.zeros(len(df_list))
#year_begin = 2025
year_begin = 2010
year_end = 2025
month_begin = 1
month_end = 12

for year_ in range(year_begin,year_end+1):
    for month_ in range(month_begin,month_end+1):
# for year_ in range(year_end,year_begin-1,-1):
#     for month_ in range(month_end,month_begin-1,-1):
        data = read_nc_year_month(year_,month_)
        if data is None:
            continue
        else:
            # t2m_month = data.variables['t2m'][:,:]
            lat_list = data.variables['latitude'][:]
            lon_list = data.variables['longitude'][:]
            #print(f"Reading t2m done")
            #print(f"lon_list: {lon_list}")
            #print(f"lat_list: {lat_list}")

            continue_flag = False
            for i in range(len(df_list)):
                date_temp = f"{year_}-{month_:02d}-1"
                if(month_ == 2 and (year_%4 == 0 and year_%100 != 0) or (year_%400 == 0)):
                    date_temp2 = f"{year_}-{month_:02d}-{month_day_list_4[int(month_)-1]}"
                    print(f"Leap year: {year_}-{month_:02d}-{month_day_list_4[int(month_)-1]}")
                else:
                    date_temp2 = f"{year_}-{month_:02d}-{month_day_list[int(month_)-1]}"
                if(is_in_days(date_temp,df_list[i][1])):
                    continue_flag = True
                elif(is_in_days(date_temp2,df_list[i][1])):
                    continue_flag = True
                # date_temp2 = f"{year_}-{month_:02d}-{month_day_list[int(month_)-1]}"
                # if(is_in_days(date_temp,df_list[i][1])):
                #     continue_flag = True
                # elif(is_in_days(date_temp2,df_list[i][1])):
                #     continue_flag=True
            if(continue_flag == True):
                t2m_large = data.variables['t2m'][:,:,:]
                # for day_ in range(month_day_list[month_-1],0,-1):
                # for day_ in range(1,month_day_list[month_-1]+1):
                if(month_ == 2 and (year_%4 == 0 and year_%100 != 0) or (year_%400 == 0)):
                    range_of_for = month_day_list_4[month_-1]
                else:
                    range_of_for = month_day_list[month_-1]
                for day_ in range(1,range_of_for+1):
                    #check t2m read is nessesary or not by checking df_list
                    nc_date = f"{year_}-{month_:02d}-{day_:02d}"
                    #for i in range(len(df_list)):
                        #if is_in_days(nc_date, df_list[i][1]):
                            #continue_flag = True
                            #break
                        #else:
                            # continue_flag = False
                           # continue
                    if continue_flag:
                        #day+hour-1
                        hr_sum = np.zeros(len(df_list))
                        hr_max = np.zeros(len(df_list))
                        hr_min = np.zeros(len(df_list))
                        for hr in range(0,24):
                            #check is in days or not
                            t2m = None
                            pass_hr = True
                            for i in range(len(df_list)):
                                if is_in_days(nc_date, df_list[i][1]):
                                    t2m = t2m_large[(day_-1)*24+hr,:,:]
                                    # continue_flag = True
                                    pass_hr = False
                                    break
                                else:
                                    continue
                            if(pass_hr == True):
                                print("Pass " + str(nc_date)+"-"+str(hr))
                            else:
                                print(f"Processing date: {nc_date}-{hr} ")
                            if(hr == 0):
                                hr_max = np.zeros(len(df_list))
                                hr_min = np.zeros(len(df_list))
                            for i in range(len(df_list)):
                                if is_in_days(nc_date, df_list[i][1]):
                                    temperature = read_nc_from_list(t2m, lon_list, lat_list, df_list[i])
                                    #hr_sum+=temperature
                                    hr_sum[i]+=temperature
                                    if(hr == 0):
                                        hr_max[i] = temperature
                                        hr_min[i] = temperature
                                    else:
                                        if(temperature>hr_max[i]):
                                            hr_max[i]=temperature
                                        if(temperature<hr_min[i]):
                                            hr_min[i]=temperature



                                    #index_acc_day = int(temperature_previous_index[i])
                                    #temperature_previous_180days[i,index_acc_day]= temperature
                                    #temperature_previous_index[i] += 1
                            #try 
                            del t2m
                        hr_avg2 = (hr_max+hr_min)/2
                        hr_avg = hr_sum/24
                        for i in range(len(df_list)):
                            index_acc_day = int(temperature_previous_index[i])
                            if(hr_avg[i]>0.01):
                                temperature_previous_180days[i,index_acc_day]= hr_avg[i]
                                temperature_previous_180days_min_max[i,index_acc_day]= hr_avg2[i]
                                temperature_previous_index[i] += 1
            else:
                 print(f"Skipping   date: {date_temp}")
                 continue
            del lon_list
            del lat_list
            del t2m_large
        print(temperature_previous_index)
        del data
                        


        print(f"Data for {year_}-{month_:02d} processed")
            
#output temperature_previous_180days to csv
temperature_previous_180days_df = pd.DataFrame(temperature_previous_180days)
temperature_previous_180days_df.to_csv('temperature_previous_365days_min_max.csv', index=False)
temperature_previous_180days_df_min_max = pd.DataFrame(temperature_previous_180days_min_max)
temperature_previous_180days_df_min_max.to_csv('temperature_previous_365days_min_max.csv', index=False)
# print("Temperature previous 180 days:")
# print(temperature_previous_180days_df)


# print("Checking if date is in 180 days...")
# Example usage
# print(is_in_180days(nc_date_test, tree_date_test))

# def get_era5_data(lat, lon, start_date, end_date):
