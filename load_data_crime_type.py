#PROCESS AND LOAD DATA

import numpy as np
import pandas as pd
import time
from datetime import datetime

num_weather_labels = 6
num_temp_labels = 6
num_pcp_labels = 6
num_wdir_labels = 6
num_wnd_labels = 6
num_rh_labels = 6
num_vis_labels = 6
num_loc_x = 4
num_loc_y = 6

weather_thresholds = {}
thresh_x = []
thresh_y = []

############ METHODS THAT EXTRACT RELEVANT INFO FROM TIME STAMPS ############

def stampToHours(stamp):
    
    date_object = datetime.strptime(stamp, '%H:%M:%S')
    if date_object.minute < 30:
        return date_object.hour
    else:
        return (date_object.hour + 1) % 24
    
def stampToDate(stamp):
    
    date_object = datetime.strptime(stamp, '%Y-%m-%d')
    return int(str(date_object.month).zfill(2)+str(date_object.day).zfill(2)) 
    
def stampToWeekDay(stamp):
    
    date_object = datetime.strptime(stamp, '%Y-%m-%d')
    return date_object.weekday()

def stampToYear(stamp):
    
    date_object = datetime.strptime(stamp, '%Y-%m-%d')
    return date_object.year
    
def stampToMonth(stamp):
    
    date_object = datetime.strptime(stamp, '%Y-%m-%d')
    return date_object.month
    

####################METHODS USED FOR SEGMENTING DATA##########################

def calc_thresholds_weather(year, weather_data_processed, feature):
    threshold_arr = []
    #data = pd.Series(weather_data[weather_data['Year'] == year]['TmpF'])
    data = pd.Series(weather_data_processed[weather_data_processed['YEAR'] ==
year][feature])
    entries_per_label = len(data)/num_weather_labels
    data.sort()
    for i in range(0,num_weather_labels-1):
        
        threshold_arr.append(float(data.head((i + 1) * entries_per_label).tail(1)))
    
    return threshold_arr

def calc_thresholds_minmax(year, weather_data_processed, feature):
    threshold_arr = []
    data = pd.Series(weather_data_processed[weather_data_processed['YEAR'] == year][feature])
    data.sort()
    data_arr = data.values
    min_elem = data_arr[0]
    max_elem = data_arr[len(data_arr) - 1]
    step = (max_elem - min_elem) / num_weather_labels
    for i in range(0,num_weather_labels-1):
        threshold_arr.append(min_elem + ((i + 1) * step))
    return threshold_arr

def calc_thresholds_loc(crime_data_processed):
    data_x = pd.Series(crime_data_processed['POINT_X'])
    data_y = pd.Series(crime_data_processed['POINT_Y'])
    data_x.sort()
    data_y.sort()
    data_x_arr = data_x.values
    data_y_arr = data_y.values
    min_x = data_x_arr[0]
    min_y = data_y_arr[0]
    max_x = data_x_arr[len(data_x_arr) - 1]
    max_y = data_y_arr[len(data_y_arr) - 1]
    step_x = (max_x - min_x) / num_loc_x
    step_y = (max_y - min_y) / num_loc_y
    
    threshold_x_arr = []
    for i in range(0,num_loc_x):
        threshold_x_arr.append(min_x + ((i + 1) * step_x))
    threshold_y_arr = []
    for i in range(0,num_loc_y):
        threshold_y_arr.append(min_y + ((i + 1) * step_y))
    return threshold_x_arr,threshold_y_arr 

def assign_loc_label(x):
    ret = 0
    for i in range(num_loc_x):
        if x['POINT_X'] <= thresh_x[i]:
            ret = i 
            break
    for i in range(num_loc_y):
        if x['POINT_Y'] <= thresh_y[i]:
            ret = ret + (i * num_loc_x)
            break
    return ret

def assign_temp_label(x):
    for i in range(num_weather_labels -1):
        if x['TEMP'] < weather_thresholds[x['YEAR']][i]:
            return i
    return num_weather_labels - 1

def assign_pcp_label(x):
    for i in range(num_weather_labels -1):
        if x['PRECIP_INCHES'] < weather_thresholds[x['YEAR']][i]:
            return i
    return num_weather_labels - 1

def assign_rh_label(x):
    for i in range(num_weather_labels -1):
        if x['HUMIDITY'] < weather_thresholds[x['YEAR']][i]:
            return i
    return num_weather_labels - 1

def assign_wnd_label(x):
    for i in range(num_weather_labels -1):
        if x['WIND'] < weather_thresholds[x['YEAR']][i]:
            return i
    return num_weather_labels - 1

def assign_wdir_label(x):
    for i in range(num_weather_labels -1):
        if x['WIND_DIR'] < weather_thresholds[x['YEAR']][i]:
            return i
    return num_weather_labels - 1

def assign_vis_label(x):
    for i in range(num_weather_labels -1):
        if x['VISIBILITY'] < weather_thresholds[x['YEAR']][i]:
            return i
    return num_weather_labels - 1

def assign_ucr_label(x):
    if x['CRIME_TYPE'] == 600:
        if x['CRIME_DESC'] == 'Thefts':
            return 600
        return 800
    return x['CRIME_TYPE']



 ########## DATA PROCESSING ############

init_time = time.time()

print "Reading crime data"
crime_data = pd.read_csv('data/police_inct.csv', low_memory=False)
crime_data_processed = pd.DataFrame()


#CREATE NEW DATAFRAME WITH DESIRED FEATURES
print "Applying new features"
crime_data_processed['POINT_X'] = crime_data['POINT_X']
crime_data_processed['POINT_Y'] = crime_data['POINT_Y']
crime_data_processed['HOUR'] = crime_data['DISPATCH_TIME'].apply(stampToHours) 
crime_data_processed['CRIME_TYPE'] = crime_data['UCR_GENERAL']
crime_data_processed['CRIME_DESC'] = crime_data['TEXT_GENERAL_CODE']
crime_data_processed['DATE'] = crime_data['DISPATCH_DATE'].apply(stampToDate)
crime_data_processed['WEEK_DAY'] = crime_data['DISPATCH_DATE'].apply(stampToWeekDay)
crime_data_processed['MONTH'] = crime_data['DISPATCH_DATE'].apply(stampToMonth)
crime_data_processed['YEAR'] = crime_data['DISPATCH_DATE'].apply(stampToYear)


#DROP INSTANCES WITH INCOMPLETE DATA AND DROP ENTRIES DUE TO LOCATION ERROR
print "Dropping erroneous values"
crime_data_processed = crime_data_processed.dropna()
crime_data_processed = crime_data_processed[crime_data_processed['POINT_X'] > -84]
crime_data_processed = crime_data_processed[crime_data_processed['POINT_Y'] > 39.2]

#N means not reported, T means trace, or very little precipitation
print "Reading weather data"
weather_data = pd.read_csv('data/weather_data.csv', skiprows = 50).replace(to_replace='N.*', value = np.nan, regex=True).replace(to_replace='T.*', value = 0, regex=True)

#Change types so we can process data later
print "Changing weather feature types"
weather_data['TmpF'] = weather_data['TmpF'].astype(float)
weather_data['Wind'] = weather_data['Wind'].astype(float)
weather_data['WDir'] = weather_data['WDir'].astype(float)
weather_data['RH'] = weather_data['RH'].astype(float)
weather_data['Vis'] = weather_data['Vis'].astype(float)
weather_data['CC'] = weather_data['CC'].astype(float)
weather_data['PcpIn'] = weather_data['PcpIn'].astype(float)


#CREATE NEW DATAFRAME WITH DESIRED FEATURES
print "Adding new weather features"
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
print "Dropping bad temp data"
weather_data_processed = weather_data_processed.dropna(how='any',subset=['TEMP','WIND','WIND_DIR','HUMIDITY','VISIBILITY','CLOUD_COVER','PRECIP_INCHES'])

#MERGE BOTH WEATHER AND CRIME DATA INTO ONE DATAFRAME
print "Merging data"
data_processed = pd.merge(crime_data_processed, weather_data_processed, how = 'outer')

#ESSENTIALLY GETS RID OF YEARS THAT ARE NOT 2009-2013 AS WELL AS DATA FOR WHICH TEMPERATURE IS MISSING
print "Dropping more bad temp stuff"
data_processed = data_processed.dropna(how='any', subset=['TEMP', 'POINT_X'])

##################BUILD LABELLED FEATURES#########################

print "Adding labelled features for:"

#TEMP
print "Temp"
num_weather_labels = num_temp_labels
for year in range(2009, 2014):
    weather_thresholds[year] = calc_thresholds_minmax(year,weather_data_processed, 'TEMP')
data_processed['TEMP_LABELED'] = data_processed.apply(assign_temp_label, axis = 1)

#PCP
print "Precipitation"
num_weather_labels = num_pcp_labels
for year in range(2009,2014):
    weather_thresholds[year] = calc_thresholds_minmax(year,weather_data_processed,'PRECIP_INCHES')
data_processed['PRECIP_LABELLED'] = data_processed.apply(assign_pcp_label,axis=1)

#HUMIDITY
print "Humidity"
num_weather_labels = num_rh_labels
for year in range(2009,2014):
    weather_thresholds[year] = calc_thresholds_minmax(year,weather_data_processed,'HUMIDITY')
data_processed['HUMIDITY_LABELLED'] = data_processed.apply(assign_rh_label,axis=1)

#WIND
print "Wind"
num_weather_labels = num_wnd_labels
for year in range(2009,2014):
    weather_thresholds[year] = calc_thresholds_minmax(year,weather_data_processed,'WIND')
data_processed['WIND_LABELLED'] = data_processed.apply(assign_wnd_label,axis=1)

#WIND_DIR
print "Wind direction"
num_weather_labels = num_wdir_labels
for year in range(2009,2014):
    weather_thresholds[year] = calc_thresholds_minmax(year,weather_data_processed,'WIND_DIR')
data_processed['WIND_DIR_LABELLED'] = data_processed.apply(assign_wdir_label,axis=1)

#VIS
print "Visibility"
num_weather_labels = num_vis_labels
for year in range(2009,2014):
    weather_thresholds[year] = calc_thresholds_minmax(year,weather_data_processed,'VISIBILITY')
data_processed['VISIBILITY_LABELLED'] = data_processed.apply(assign_vis_label,axis=1)

#LOC
print "Location"
a,b = calc_thresholds_loc(crime_data_processed)
thresh_x = a
thresh_y = b
data_processed['LOC_LABELLED'] = data_processed.apply(assign_loc_label,axis=1)

#Split thefts
print "Splitting theft types"
data_processed['CRIME_TYPE'] = data_processed.apply(assign_ucr_label,axis=1)

#print type(data_processed['WIND'][0])
print data_processed[:30]
print ("runtime is: %f" % (time.time() - init_time))

#STORE DATA
data_processed.to_pickle('data/data_processed.pkl')
