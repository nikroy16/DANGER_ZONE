import numpy as np
import pandas as pd



#PRODUCES RATE DATA SETS (for parts 2 and 3)



data_processed = pd.read_pickle('data/data_processed.pkl')

######################################################################################
#sample 1 - 2 features: month and temp

rate_df = pd.DataFrame(columns = ['YEAR', 'MONTH', 'TEMP_LABELED', 'RATE'])
counter = 0
for year in range(2009, 2014):
    num_entries_in_year = len(data_processed[data_processed['YEAR'] == year])
    for i in range(24):
        for j in range(6):
            
            print '\rProgress [%d%%]'%((counter * 100)/(5 * 24 * 6)),
            counter += 1
            entries = data_processed[data_processed['YEAR'] == year][data_processed['MONTH'] == i][data_processed['TEMP_LABELED'] == j]
            
            h = pd.Series({'YEAR':year,'MONTH':i,'TEMP_LABELED':j,'RATE':float(len(entries))/float(num_entries_in_year)})
            rate_df = rate_df.append(h, ignore_index = True)
            
rate_df.to_pickle('data/month_temp_rate.pkl')


#########################################################################################
#sample 2 - 3 features: hour, wind and week day

rate_df = pd.DataFrame(columns = ['YEAR', 'HOUR', 'WIND_LABELLED', 'WEEK_DAY', 'RATE'])
counter = 0
for year in range(2009, 2014):
    num_entries_in_year = len(data_processed[data_processed['YEAR'] == year])
    for i in range(24):
        for j in range(6):
            for k in range(7):
                print '\rProgress [%d%%]'%((counter * 100)/(5 * 24 * 6 * 7)),
                counter += 1
                entries = data_processed[data_processed['YEAR'] == year][data_processed['HOUR'] == i][data_processed['WIND_LABELLED'] == j][data_processed['WEEK_DAY'] == k]

                h = pd.Series({'YEAR':year,'HOUR':i,'WIND_LABELLED':j, 'WEEK_DAY':k,'RATE':float(len(entries))/float(num_entries_in_year)})
                rate_df = rate_df.append(h, ignore_index = True)
                
rate_df.to_pickle('data/hour_wind_day_rate.pkl')
