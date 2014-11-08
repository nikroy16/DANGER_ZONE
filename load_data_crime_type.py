#PROCESS AND LOAD DATA

import numpy as np
import pandas as pd
from datetime import datetime


num_temp_labels = 6
wind_threshold = 20.0


weather_thresholds = {}



############ METHODS THAT EXTRACT RELEVANT INFO FROM TIME STAMPS ############

def stampToHours(stamp):
    
    date_object = datetime.strptime(stamp, '%H:%M:%S')
    if date_object.minute < 30:
        return date_object.hour
    else:
        return (date_object.hour + 1) % 24
    
def stampToDate(stamp):
    
    date_object = datetime.strptime(stamp, '%Y-%m-%d')
    return str(date_object.month).zfill(2)  + str(date_object.day).zfill(2) 
    
def stampToWeekDay(stamp):
    
    date_object = datetime.strptime(stamp, '%Y-%m-%d')
    return date_object.weekday()

def stampToYear(stamp):
    
    date_object = datetime.strptime(stamp, '%Y-%m-%d')
    return date_object.year
    
def stampToMonth(stamp):
    
    date_object = datetime.strptime(stamp, '%Y-%m-%d')
    return date_object.month
    
    


#METHODS USED FOR SEGMENTING DATA

def calc_thresholds_temp(year, weather_data_processed):
    threshold_arr = []
    #data = pd.Series(weather_data[weather_data['Year'] == year]['TmpF'])
    data = pd.Series(weather_data_processed[weather_data_processed['YEAR'] == year]['TEMP'])
    entries_per_label = len(data)/num_temp_labels
    data.sort()
    for i in range(0,num_temp_labels -1):
        
        threshold_arr.append(float(data.head((i + 1) * entries_per_label).tail(1)))
    
    return threshold_arr

def assign_temp_label(x):
    for i in range(num_temp_labels -1):
        #print x['TEMP']
        if x['TEMP'] < weather_thresholds[x['YEAR']][i]:
            return i
    
    return num_temp_labels - 1




 ########## DATA PROCESSING ############





crime_data = pd.read_csv('data/police_inct.csv', low_memory=False)


crime_data_processed = pd.DataFrame()


#CREATE NEW DATAFRAME WITH DESIRED FEATURES
crime_data_processed['POINT_X'] = crime_data['POINT_X']
crime_data_processed['POINT_Y'] = crime_data['POINT_Y']
crime_data_processed['HOUR'] = crime_data['DISPATCH_TIME'].apply(stampToHours) 
crime_data_processed['CRIME_TYPE'] = crime_data['UCR_GENERAL']
crime_data_processed['DATE'] = crime_data['DISPATCH_DATE'].apply(stampToDate)
crime_data_processed['WEEK_DAY'] = crime_data['DISPATCH_DATE'].apply(stampToWeekDay)
crime_data_processed['MONTH'] = crime_data['DISPATCH_DATE'].apply(stampToMonth)
crime_data_processed['YEAR'] = crime_data['DISPATCH_DATE'].apply(stampToYear)


#DROP INSTANCES WITH INCOMPLETE DATA AND DROP ENTRIES DUE TO LOCATION ERROR
crime_data_processed = crime_data_processed.dropna()
crime_data_processed = crime_data_processed[crime_data_processed['POINT_X'] > -84]

#N means not reported, T means trace, or very little precipitation
weather_data = pd.read_csv('data/weather_data.csv', skiprows = 50).replace(to_replace='N.*', value = np.nan, regex=True).replace(to_replace='T.*', value = 0, regex=True)

#Change types so we can process data later
weather_data['TmpF'] = weather_data['TmpF'].astype(float)
weather_data['Wind'] = weather_data['Wind'].astype(float)
weather_data['WDir'] = weather_data['WDir'].astype(float)
weather_data['RH'] = weather_data['RH'].astype(float)
weather_data['Vis'] = weather_data['Vis'].astype(float)
weather_data['CC'] = weather_data['CC'].astype(float)
weather_data['PcpIn'] = weather_data['PcpIn'].astype(float)


#CREATE NEW DATAFRAME WITH DESIRED FEATURES
weather_data_processed = pd.DataFrame()
weather_data_processed['DATE'] = weather_data['Date'].apply(stampToDate)
weather_data_processed['HOUR'] = weather_data['Time'].apply(stampToHours) 
weather_data_processed['TEMP'] = weather_data['TmpF']
weather_data_processed['WIND'] = weather_data['Wind']
weather_data_processed['WIND_DIR'] = weather_data['WDir']
weather_data_processed['S_RAIN'] = weather_data['S_Rain']
weather_data_processed['S_SNOW'] = weather_data['S_Snow']
weather_data_processed['HUMIDITY'] = weather_data['RH']
weather_data_processed['VISIBILITY'] = weather_data['Vis']
weather_data_processed['CLOUD_COVER'] = weather_data['CC']
weather_data_processed['PRECIP_INCHES'] = weather_data['PcpIn']
weather_data_processed['YEAR'] = weather_data['Date'].apply(stampToYear)



#DROP WEATHER DATA THAT IS MISSING TEMP
weather_data_processed = weather_data_processed.dropna(how='all', subset=['TmpF'])





#MERGE BOTH WEATHER AND CRIME DATA INTO ONE DATAFRAME
data_processed = pd.merge(crime_data_processed, weather_data_processed, how = 'outer')

#ESSENTIALLY GETS RID OF YEARS THAT ARE NOT 2009-2013 AS WELL AS DATA FOR WHICH TEMPERATURE IS MISSING
data_processed = data_processed.dropna(how='any', subset=['TEMP', 'POINT_X'])



for year in range(2009, 2014):
    weather_thresholds[year] = calc_thresholds_temp(year, weather_data_processed)


data_processed['TEMP_LABELED'] = data_processed.apply(assign_temp_label, axis = 1)
data_processed['WIND_BINARY'] = data_processed['WIND'] > wind_threshold
#print type(data_processed['WIND'][0])

#STORE DATA
data_processed.to_pickle('data/data_processed.pkl')


   
    
    


    











