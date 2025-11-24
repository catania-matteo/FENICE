# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:08:06 2023

@author: MC
"""
# usare omeof_snap_clust env


import os
import logging
import pandas as pd
#import oemof.solph as solph
#import custom
# import numpy as np
#from oemof.tools import logger
#from pyomo import environ as po
# from pprint import pprint
# from numpy import genfromtxt
# from matplotlib import pyplot as plt

import datetime
# import fer_constraint
# import w_constr
# import pv_constr_2

import numpy as np

import pickle
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import (
    pairwise_distances,
    pairwise_distances_argmin,
)

import tsam.timeseriesaggregation as tsam


start_date = datetime.datetime(2013, 1, 1)

#### options #####

list_number_typ_days = [24,36]#[4,12,24,36,52,104]
list_delta_year = [5]#[10,5,3,2,1]

list_method_clust = ["k_medoids", "k_means"] #["k_means"]#["k_medoids", "k_means"]##["k_medoids", "k_means"] 

folder_name = "result_clustering"
os.makedirs(folder_name, exist_ok=True)
R = {'r01', 'r02', 'r03', 'r04','r05','r06','r07','r08','r09','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20'}


weight_dict={}
# ACTIVATE if you want weight or peaks in the clustering
##value_weight = 10
# for nam in profiles_names:
#     weight_dict[nam] = 1


# list_columns_to_add_weight = []
# list_columns_to_add_peaks = []
# names= ['_solar_pv']#['_wind','_solar_pv','_load_el_net']#['_solar_pv', '_wind','_wind_off','_load_el_net']



# for nam in names:
#     for i in R:
#         label_col = i + nam
#         list_columns_to_add_peaks.append(label_col)
#         list_columns_to_add_weight.append(label_col)


# for nam in list_columns_to_add_weight:
#     weight_dict[nam] = 100

for delta in list_delta_year:
    for number_typ_days in list_number_typ_days:
        for method_clust in list_method_clust:
            InputData = 'Input-V36-Mononode.xlsx'
            #%% Clustering
            InputData = {}
            for year in range(2025, 2056, delta):
                InputData[year] = f"Year Conf/Delta {delta}/{year}.xlsx"

            # InputData1 = 'modified_2025.xlsx'
            # InputData2 = 'modified_2030.xlsx'
            # InputData3 = 'modified_2035.xlsx'
            # InputData4 = 'modified_2040.xlsx'
            # InputData5 = 'modified_2045.xlsx'
            # InputData6 = 'modified_2050.xlsx'
            # InputData7 = 'modified_2055.xlsx'
            
            merged_profiles = pd.DataFrame()
            merged_repetition_per_hour = []
            cluster_order_days_updated_dict = {}
            primi_giorni_clusters_dict = {}
            primi_giorni_clusters_overall = []
            cluster_order_to_updated_dict={}
            concatenated_datetime_index_all_test= 0#pd.date_range()
            
            list_InputData =[]
            for key, value in InputData.items():
                list_InputData.append(value)
            
            folder_name = f"result_clustering/Delta {delta}"
            os.makedirs(folder_name, exist_ok=True)
            
            
            #list_InputData = [InputData1,InputData2,InputData3,InputData4,InputData5,InputData6,InputData7]
            
            
            
            h= -1
            
            
            for InputData in list_InputData:
                filename = os.path.join(os.path.dirname(__file__), InputData)
                xls = pd.ExcelFile(filename)
                timeseries = xls.parse('Sheet1')

                # extract the names in the first row of the sheet
                profiles_names = list(timeseries.columns)
                profiles_names = profiles_names[1:]
                # clustering settings
                clustering_options = {
                    # Clustering algorithm options
                    'profiles': profiles_names,
                    # Clustering periods options
                    # Specify the number of clusters (typical periods)
                    'n_clusters': number_typ_days,
                    # period length in hours
                    'period_length': 24,        # lenght in hours of the typical period (e.g. 1 day --> 24 h)
                    'dt': 1,                 # time resolution (hour or fraction of h)
                    # NOTE: the time resolution (dt) will be overwritten when reading
                    # the excel file containing the profiles if a DateTime column is provided
                }
                #%% TSAM VERSION

                #raw = pd.read_csv('testdata.csv', index_col = 0)
                raw = pd.read_excel(InputData, index_col = 0)
                raw = raw.round(4)
                #raw= raw.drop(columns=['Unnamed: 0'])
                datetime_index = pd.date_range(start='2023-01-01 00:00:00', end='2023-12-31 23:00:00', freq='H')
                raw.index= datetime_index

                aggregation = tsam.TimeSeriesAggregation(raw,
                  noTypicalPeriods = clustering_options['n_clusters'],
                  hoursPerPeriod = clustering_options['period_length'],
                  segmentation = True,
                  noSegments = 24,
                  #representationMethod = "distributionAndMinMaxRepresentation",
                  distributionPeriodWise = False,
                  clusterMethod = method_clust,
                  #extremePeriodMethod= "new_cluster_center", #"replace_cluster_center",  #"new_cluster_center",
                  #addMeanMax = list_columns_to_add_peaks,
                  #addMeanMin = list_columns_to_add_peaks
                  #weightDict= weight_dict,
                )

                a = aggregation.clusterOrder



                cluster_order_to_oemof = []
                b = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
                for x in range(len(aggregation.clusterOrder)):
                    for i in range(24):
                      adds = b[i] + 24 * aggregation.clusterOrder[x]
                      cluster_order_to_oemof.append(adds)  
                    
                 
                repetition_per_hour_oemof= {}
                for j in range(aggregation.noTypicalPeriods):
                    
                    repetition_per_hour_oemof[j] = aggregation.clusterPeriodNoOccur[j]

                typPeriods = aggregation.createTypicalPeriods()

                #typPeriods.to_csv('typperiods.csv')

                # count=0
                # for h in range(len(a)):
                #     if a[h] == 11:
                #          count= count + 1

                first_index = {}
                ordine_clusters = []
                primi_giorni_clusters = []
                #first_index = OrderedDict()
                for i, d in enumerate(aggregation.clusterOrder):
                    if d not in first_index:
                        first_index[d] = i
                        ordine_clusters.append(d)
                        primi_giorni_clusters.append(i)

                d= aggregation.clusterPeriodNoOccur
                new_index={}
                list_days =[]
                dict_conv_n_cluster ={}
                repetition_per_hour= []
                i=0
                for key, value in first_index.items():
                    dict_conv_n_cluster[key]= i
                    new_index[value] = key
                    #dict_conv_n_cluster
                    list_days.append(key)
                    i=i+1


                cluster_order_days_updated = []
                for i in range(len(aggregation.clusterOrder)):
                    step1 = aggregation.clusterOrder[i]
                    value = dict_conv_n_cluster[step1]
                    cluster_order_days_updated.append(value)

                cluster_order_to_updated = []
                b = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
                for x in range(len(cluster_order_days_updated)):
                    for i in range(24):
                      adds = b[i] + 24 * cluster_order_days_updated[x]
                      cluster_order_to_updated.append(adds)  




                    
                for key, value in new_index.items():
                    c = aggregation.clusterPeriodNoOccur[value]
                    for f in range(24):
                        repetition_per_hour.append(c) 
                    

                # Your custom order for the 'index' column
                custom_order = list_days
                list_hours = []
                for x in range(len(list_days)):
                    for i in range(24):
                      adds1 = b[i] + 24 * list_days[x]
                      list_hours.append(adds1)
                    
                

                #df_sorted = typPeriods.sort_index(key=lambda x: x.map(dict(zip(custom_order, range(len(custom_order))))))
                
                #df_sorted = typPeriods.sort_index(level=1, key=lambda x: x.map(dict(zip(custom_order, range(24)))))
                
                
                typPeriods= typPeriods.reset_index()
                df_sorted = typPeriods.sort_index( key=lambda x: x.map(dict(zip(list_hours,range(len(list_hours))))))
                
                profiles = df_sorted.reset_index()





                # Define the start date and first_index
                start_date = '2022-01-01'

                new_list = [j  for j in primi_giorni_clusters]
                first_index1 = new_list
                # convert first_index to a list of integers
                first_index_int = [int(x) for x in first_index1]

                # create a list of dates using the start date and the list of integers
                dates = [pd.Timestamp(start_date) + pd.DateOffset(days=i) for i in first_index_int]


                # Calculate the date for each value in first_index
                # dates = [pd.Timestamp(start_date) + pd.DateOffset(days=i) for i in first_index]

                # Format the dates as strings in the desired format
                dates = [d.strftime('%Y-%m-%d') for d in dates]

                # Define the start and end dates for the date range
                year_to_append= 2022 + h +1
                start_date = f'{year_to_append}-01-01 00:00:00'
                end_date = f'{year_to_append}-12-31 23:59:59'

                # Define the 8 non-consecutive days you want to include in the date range
                #days = [pd.to_datetime(d) for d in ['2022-01-01', '2022-02-01', '2022-03-01','2022-04-01', '2022-05-01','2022-06-01', '2022-07-01','2022-08-01','2022-09-01','2022-10-01','2022-11-01','2022-12-01']]
                days = [pd.to_datetime(d) for d in dates]

                # Create the list of all hours of the day
                hours = [pd.to_datetime(f"{d.date()} {h:02d}:00:00") for d in days for h in range(24)]
                # get the last timestamp in the list and add one hour
                last_timestamp = hours[-1]
                next_timestamp = last_timestamp + pd.Timedelta(hours=1)

                # append the next timestamp to the list of hours
                hours.append(next_timestamp)

                # Concatenate the days and hours lists
                days_hours = days + hours

                # Create the date range
                date_range = pd.date_range(start_date, end_date, freq='H')

                # Filter the date range to only include the 8 non-consecutive days and all hours
                date_range = date_range[date_range.isin(days_hours)]

                # datetime_index_all = pd.date_range(
                #                 '1/1/2013', periods = number_timesteps*2, freq='H')
                datetime_index_all = date_range#raw = pd.read_csv('testdata.csv', index_col = 0)
                
                
                
                
                
                if number_typ_days==365:
                    profiles =timeseries
                
                
                
                
                
                
                
                merged_profiles = pd.concat([merged_profiles, profiles], ignore_index=True)
                merged_profiles.reset_index(drop=True, inplace=True)
                
                
                
                
                merged_repetition_per_hour.append(repetition_per_hour)
                
                
                
                
                h= h+1
                cluster_order_days_updated_dict[h]= cluster_order_days_updated
                primi_giorni_clusters_dict[h] = primi_giorni_clusters

                
                cluster_order_to_updated_dict[h]=cluster_order_to_updated
                
                
                if concatenated_datetime_index_all_test ==0:
                    concatenated_datetime_index_all=datetime_index_all
                    concatenated_datetime_index_all_test=1
                else:
                    
                    concatenated_datetime_index_all.append(datetime_index_all) 
                    
                print(f'{number_typ_days}_{method_clust}   DONE')
                #logger.define_logging()
                logging.info(f'{number_typ_days}_{method_clust} {InputData}  DONE')


            # Save the data to a file
            filename = f'{folder_name}/datetime_index_all_tsam_{method_clust}_{number_typ_days}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(concatenated_datetime_index_all, f)

            filename = f'{folder_name}/repetition_per_hour_tsam_{method_clust}_{number_typ_days}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(merged_repetition_per_hour, f)
                
            filename = f'{folder_name}/\primi_giorni_clusters_tsam_{method_clust}_{number_typ_days}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(primi_giorni_clusters_dict, f)    
                
                
            filename = f'{folder_name}/final_list1_tsam_{method_clust}_{number_typ_days}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(cluster_order_to_updated_dict, f)  

                
            filename = f'{folder_name}/profiles_tsam_{method_clust}_{number_typ_days}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(merged_profiles, f)  
                

                
            filename = f'{folder_name}/var_M_tsam_{method_clust}_{number_typ_days}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(cluster_order_days_updated_dict, f)    





