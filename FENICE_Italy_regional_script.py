# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:41:49 2024

@author: Matteo Catania
"""

import os
import logging
import pandas as pd
import oemof.solph as solph
from oemof.tools import logger
from oemof.solph import processing, views
import numpy as np
from pyomo.core import Binary
from pyomo.core import Constraint
from pyomo.core import Expression
from pyomo.core import NonNegativeReals
from pyomo.core import Set
from pyomo.core import Var
from pyomo.core.base.block import SimpleBlock
from pyomo import environ as po
import sys
sys.path.append('C:/Users/Matteo Catania/Desktop/Esempio_kMeans')

import pickle
import matplotlib.pyplot as plt

InputData = 'Input-Italy-20R.xlsx'#'Input-V40.xlsx'
filename = os.path.join(os.path.dirname(__file__), InputData,)
xls = pd.ExcelFile(filename)

clustering_options = {
    'n_clusters': 24,
    'period_length': 24,        # lenght in hours of the typical period (e.g. 1 day --> 24 h)
    'dt': 1,                 # time resolution (hour or fraction of h)
}

delta= 5
method_clust ='k_medoids'
number_typ_days= clustering_options['n_clusters']

dict_number_year = {10: 4,
                    5: 7,
                    3: 11,
                    2: 16,
                    1: 31
                    }


R = {'r01', 'r02', 'r03', 'r04','r05','r06','r07','r08','r09','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20'}

number_periods = 7
delta_year_periods = 5
number_timesteps = 24*clustering_options['n_clusters']


def series_variable_costs(series, n_timesteps):
    list_output=[]
    for i in range(len(series)):
        value = [series[i]]*n_timesteps
        list_output.extend(value)
    return list_output


blend_max_percentage ={
    "000": [0,0,0,0,0,0,0],
    "020":[0,0.1,0.2,0.2,0.2,0.2,0.2],
    "100":[0,0.1,0.2,0.5,1,1,1]}

hydrogen_invest ={
   "0": False,
   "1":True}

ee_grid_invest ={
   "0": False,
   "1": True}

co2_invest ={
   "0": False,
   "1": True}

cumulative_co2_options ={
    "0": 10000000,
    "1":  5640000,
    "2":  2290000,
    "3": 0
    
    }

discount_rate = 0.05
multiplier_excess = 2

cases=  [ '100_1_1_1_3',
      ] #'100_1_1_1_2','100_1_1_1_1'


base_folder = "Results"
if not os.path.exists(base_folder):
    os.makedirs(base_folder)

for case in cases:    
    
    blend_max_percentage_case= blend_max_percentage[case.split('_')[0]]
    blend_folder= case.split('_')[0]
    hydrogen_invest_case= hydrogen_invest[case.split('_')[1]]
    ee_grid_invest_case= ee_grid_invest[case.split('_')[2]]
    co2_grid_invest_case= co2_invest[case.split('_')[3]]
    comulative_case= cumulative_co2_options[case.split('_')[4]]
    LF_grid_case = True
    
    case_folder = os.path.join(base_folder, f"Blend_{blend_folder}_h2Inv_{hydrogen_invest_case}_EEInv_{ee_grid_invest_case}_invco2pip{co2_grid_invest_case}_co2cum_{comulative_case}_discount_{discount_rate}_mulex_{multiplier_excess}")
    if not os.path.exists(case_folder):
        os.makedirs(case_folder)
    
    folder_name_input = f"result_clustering/Delta {delta}"    
    
    filename1 = f'{folder_name_input}\profiles_tsam_{method_clust}_{number_typ_days}.pkl'
    # Load the data from the file
    with open(filename1, 'rb') as f:
        profiles = pickle.load(f)
    
    filename2 = f'{folder_name_input}/final_list1_tsam_{method_clust}_{number_typ_days}.pkl'
    # Load the data from the file
    with open(filename2, 'rb') as f:
        final_list1 = pickle.load(f)

    filename3 = f'{folder_name_input}\primi_giorni_clusters_tsam_{method_clust}_{number_typ_days}.pkl'
    # Load the data from the file
    with open(filename3, 'rb') as f:
        primi_giorni_clusters = pickle.load(f)

    filename4 = f'{folder_name_input}/repetition_per_hour_tsam_{method_clust}_{number_typ_days}.pkl'
    # Load the data from the file
    with open(filename4, 'rb') as f:
        repetition_per_hour = pickle.load(f)

    filename5 = f'{folder_name_input}\datetime_index_all_tsam_{method_clust}_{number_typ_days}.pkl'
    # Load the data from the file
    with open(filename5, 'rb') as f:
        datetime_index_all = pickle.load(f)
        
    filename5 = f'{folder_name_input}/var_M_tsam_{method_clust}_{number_typ_days}.pkl'
    # Load the data from the file
    with open(filename5, 'rb') as f:
        var_M = pickle.load(f)   
    
    
    
    number_years= dict_number_year[delta]
    number_periods= number_years
    
    if number_typ_days == 365:
        #profiles = timeseries.drop('Date',axis=1)
        repetition_per_hour_model = [1]* 8760*number_years
        datetime_index_all = pd.date_range('1/1/2022', periods = 8760*number_years+1, freq='H')
        #datetime_index_all = datetime_index_all.append(pd.date_range('1/1/2023',periods = 1, freq ='H'))
        var_M = list(range(365*number_years))
        primi_giorni_clusters= list(range(365*number_years))
        final_list_model= list(range(8760*number_years))
    
    # %%         nodes creation                  
    def nodes_from_excel(filename, clustered_profiles):
    

    
                xls = pd.ExcelFile(filename)
                timeseries = clustered_profiles
                demand_nominal_value_r = xls.parse ('demand_nominal_value',header=0 , index_col = 0 )
                line_matrix_constraint = xls.parse('line_matrix_con', header=0, index_col=0)
                series_ep_cost = xls.parse('series_ep_cost',header =0, index_col=0)
                lifetime = xls.parse('lifetime',header =0, index_col=0)
                fixed_cost = xls.parse('fixed_cost',header =0, index_col=0)
                series_var_cost = xls.parse('source_ costs',header =0, index_col=0)                
                existing_capacities = xls.parse('existing transformer_cap_GW', header =0, index_col=0)
                age_values = xls.parse('age_existing', header =0, index_col=0)
                overall_maximum_values = xls.parse('overall_maximum',header =0, index_col=0)
                #fixed_costs_values = xls.parse('fixed_cost',header =0)
                storage_char_cap = xls.parse('storage char power capacity',header =0, index_col=0)
                storage_dischar_cap = xls.parse('storage dis power capacity',header =0, index_col=0)
                storage_ene_cap = xls.parse('storage energy capacity',header =0, index_col=0)
                supply_gas = xls.parse('supply estero gas',header =0, index_col=0)
                supply_LF = xls.parse('supply LF',header =0, index_col=0)
                supply_EE = xls.parse('supply estero EE',header =0, index_col=0)
                prod_naz_gas = xls.parse('Prod naz gas',header =0, index_col=0)                
                disp_biomass = xls.parse('Biomass',header =0, index_col=0)
                co2_export = xls.parse('co2_export_reg',header =0, index_col=0)
            
                R = {'r01', 'r02', 'r03', 'r04','r05','r06','r07','r08','r09','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20'}                  
                noded = {}    
                           
                bus=xls.parse('buses')
                for i, b in bus.iterrows():
                    for r in R:
                        noded[r + b['bus_suffix']] = solph.Bus(label = r + b['bus_suffix'])
                        
                        
                    if b['excess'] == 1:
                            for r in R:
                              label = r + b['bus_suffix'] + '_excess'
                              noded[label] = solph.components.Sink(label=label, inputs={
                              noded[ r + b['bus_suffix']]: solph.Flow(
                                  
                                  emission_constraint = b['emision constraint']
                                  )})
                    if b['excess'] == 2:
                       if co2_grid_invest_case == True:
                           for r in R:
                            if co2_export.loc[r,'export'] == 2:
                                label = r + b['bus_suffix'] + '_excess_Ravenna'
                                noded[label] = solph.components.Sink(label=label, inputs={
                                noded[ r + b['bus_suffix']]: solph.Flow(
                                    variable_costs= 0,
                                    emission_constraint = b['emision constraint']
                                    )})
                                label = r + b['bus_suffix'] + '_excess_shipping'
                                noded[label] = solph.components.Sink(label=label, inputs={
                                noded[ r + b['bus_suffix']]: solph.Flow(
                                    variable_costs= [x * multiplier_excess for x in series_variable_costs(series_var_cost[b['bus_suffix'] + '_excess_shipping'], number_timesteps)],
                                    )})
                                
                            if co2_export.loc[r,'export'] == 1:
                                label = r + b['bus_suffix'] + '_excess_shipping'
                                noded[label] = solph.components.Sink(label=label, inputs={
                                noded[ r + b['bus_suffix']]: solph.Flow(
                                    variable_costs= [x * multiplier_excess for x in series_variable_costs(series_var_cost[b['bus_suffix'] + '_excess_shipping'], number_timesteps)],
                                    )})
                       else:
                            for r in R:#if co2_export.loc[r,'export'] == 2:
                                label = r + b['bus_suffix'] + '_excess_Ravenna'
                                noded[label] = solph.components.Sink(label=label, inputs={
                                noded[ r + b['bus_suffix']]: solph.Flow(
                                    variable_costs= b['variable_costs_1_[M€/ktonCO2]'],
                                    emission_constraint = b['emision constraint']
                                    )})
                                label = r + b['bus_suffix'] + '_excess_shipping'
                                noded[label] = solph.components.Sink(label=label, inputs={
                                noded[ r + b['bus_suffix']]: solph.Flow(
                                    variable_costs= [x * multiplier_excess for x in series_variable_costs(series_var_cost[b['bus_suffix'] + '_excess_shipping'], number_timesteps)],
                                    )})
                            
    
                
                sources=xls.parse('sources')
                for i, b in sources.iterrows():
                    for r in R:    
                        if b['Inv'] == 1:
                           label= r + b['source_suffix']
                           noded[label] = solph.components.Source(label = label, 
                           outputs={
                           noded[r + b['bus_out_suffix']] : solph.Flow(
                               variable_costs=b['variable_costs'],
                               emission = b['emission_factor [kton CO2/GWh]'],
                               inv_ramp= b['inv_ramp'],
                                investment=solph.Investment( 
                                existing= existing_capacities.loc[r,b['source_suffix']],
                                overall_maximum= overall_maximum_values.loc[r,b['source_suffix']],
                                maximum = overall_maximum_values.loc[r,b['source_suffix']]/5,#number_periods,
                                ep_costs = series_ep_cost[b['source_suffix']],
                                fixed_costs = fixed_cost[b['source_suffix']].loc[0],
                                lifetime =lifetime[b['source_suffix']].loc[1],                                
                                age= age_values.loc[r,b['source_suffix']],                                
                                interest_rate =b['interest rate'],
                                cost_decrease =b['cost_decrease'] ,),                                
                                fixed=b['fixed'],
                                fix= (timeseries[label] if b['fixed'] == True else 1)
                               )}
                           )
                           
                        elif b['Inv'] == 'supply gas':
                                  if supply_gas.loc[r,'GWh/h']  > 0 :
                                      label= r + b['source_suffix']
                                      noded[label] = solph.components.Source(label = label, 
                                      outputs={
                                      noded[ r + b['bus_out_suffix']] : solph.Flow(
                                          variable_costs= series_variable_costs(series_var_cost[b['source_suffix']], number_timesteps),
                                          nominal_value = supply_gas.loc[r,'GWh/h'],
                                          gas_grid_par=supply_gas.loc[r,'grid_par'],
                                          supply_2050_2055 = b['supply_2050_2055'],                                        
                                          )
                                      }
                                      )
                        elif b['Inv'] == 'supply LF':
                                  if supply_LF.loc[r,'import']  > 0 :
                                      label= r + b['source_suffix']
                                      noded[label] = solph.components.Source(label = label, 
                                      outputs={
                                      noded[ r + b['bus_out_suffix']] : solph.Flow(
                                          variable_costs= series_variable_costs(series_var_cost[b['source_suffix']], number_timesteps),
                                          supply_2050_2055 = b['supply_2050_2055'],
                                          emission_constraint = b['emission constraint']
                                        
                                          )
                                      }
                                      )
                                      
                        elif b['Inv'] == 'supply h2':
                                  if supply_gas.loc[r,'h2 gas']  > 0 :
                                      label= r + b['source_suffix']
                                      noded[label] = solph.components.Source(label = label, 
                                      outputs={
                                      noded[ r + b['bus_out_suffix']] : solph.Flow(
                                          variable_costs= series_variable_costs(series_var_cost[b['source_suffix']], number_timesteps),
                                          nominal_value = supply_gas.loc[r,'h2 gas'],
                                          gas_grid_par=supply_gas.loc[r,'grid_par'],
                                          supply_2050_2055 = b['supply_2050_2055'],
                                          emission_constraint = b['emission constraint']
                                        
                                          )
                                      }
                                      )
                                      
                        elif b['Inv'] == 'supply h2 liq':
                                  if supply_gas.loc[r,'h2 liq']  > 0 :
                                      label= r + b['source_suffix']
                                      noded[label] = solph.components.Source(label = label, 
                                      outputs={
                                      noded[ r + b['bus_out_suffix']] : solph.Flow(
                                          variable_costs= series_variable_costs(series_var_cost[b['source_suffix']], number_timesteps),
                                          nominal_value = supply_gas.loc[r,'h2 liq'],
                                          supply_2050_2055 = b['supply_2050_2055'],
                                          emission_constraint = b['emission constraint']
                                        
                                          )
                                      }
                                      )
                        elif b['Inv'] == 'supply EE':
                                  if supply_EE.loc[r,'GW']  > 0 :
                                      label= r + b['source_suffix']
                                      noded[label] = solph.components.Source(label = label, 
                                      outputs={
                                      noded[ r + b['bus_out_suffix']] : solph.Flow(
                                          variable_costs=  series_variable_costs(series_var_cost[b['source_suffix']], number_timesteps),
                                          nominal_value = supply_EE.loc[r,'GW'],
                                          emission_constraint = b['emission constraint'],
                                          supply_constraint = b['supply_constraint'],
                                          supply_2050_2055 = b['supply_2050_2055'],
                                          )
                                      }
                                      )      
                          
                        elif b['Inv'] == 'fixed quantity year':
                                      label= r + b['source_suffix']
                                      noded[label] = solph.components.Source(label = label, 
                                      outputs={
                                      noded[ r + b['bus_out_suffix']] : solph.Flow(
                                          variable_costs=    series_variable_costs(series_var_cost[b['source_suffix']], number_timesteps),
                                          nominal_value = prod_naz_gas.loc[r,'nominal value'],
                                          full_load_time_max = prod_naz_gas.loc[r,'limit year'],
                                          emission_constraint = b['emission constraint']
                                        
                                          )
                                      }
                                      )
                        elif b['Inv'] == 'biomass':
                                      label= r + b['source_suffix']
                                      noded[label] = solph.components.Source(label = label, 
                                      outputs={
                                      noded[ r + b['bus_out_suffix']] : solph.Flow(
                                          variable_costs=  series_variable_costs(series_var_cost[b['source_suffix']], number_timesteps),
                                          nominal_value = disp_biomass.loc[r,'nominal value'],
                                          full_load_time_max = disp_biomass.loc[r,'limit year'],
                                          emission_constraint = b['emission constraint']
                                        
                                          )
                                      }
                                      )
                        elif b['Inv'] == 'biomass imported':
                                      label= r + b['source_suffix']
                                      noded[label] = solph.components.Source(label = label, 
                                      outputs={
                                      noded[ r + b['bus_out_suffix']] : solph.Flow(
                                          variable_costs=  series_variable_costs(series_var_cost[b['source_suffix']], number_timesteps),
                                          supply_2050_2055 = b['supply_2050_2055'],
                                          )
                                      }
                                      )
                        elif b['Inv'] == 'fixed source no inv':
                            label= r + b['source_suffix']
                            noded[label] = solph.components.Source(label = label, 
                            outputs={
                            noded[r + b['bus_out_suffix']] : solph.Flow(
                                variable_costs=b['variable_costs'],
                                emission = b['emission_factor [kton CO2/GWh]'],
                                emission_constraint = b['emission constraint'],
                                 nominal_value=1,
                                 fixed=b['fixed'],
                                 fix= (timeseries[label] if b['fixed'] == True else 1)
                            
                                # 
                                )}
                            )
                            
                        elif b['Inv'] == 'hydro ror':
                           label= r + b['source_suffix']
                           noded[label] = solph.components.Source(label = label, 
                           outputs={
                           noded[r + b['bus_out_suffix']] : solph.Flow(
                               variable_costs=b['variable_costs'],
                               emission = b['emission_factor [kton CO2/GWh]'],
                                nominal_value= storage_dischar_cap.loc[r,b['source_suffix']],
                                fixed=b['fixed'],
                                fix= (timeseries[label] if b['fixed'] == True else 1)
                               )}
                           ) 
                        elif b['Inv'] == 'waste':
                           label= r + b['source_suffix']
                           noded[label] = solph.components.Source(label = label, 
                           outputs={
                           noded[r + b['bus_out_suffix']] : solph.Flow(
                               variable_costs=b['variable_costs'],
                               emission = b['emission_factor [kton CO2/GWh]'],
                                nominal_value= 1,
                                fixed=b['fixed'],
                                fix= (timeseries[label] if b['fixed'] == True else 1)
                               )}
                           )
                        else:
                            label= r + b['source_suffix']
                            noded[label] = solph.components.Source(label = label, 
                            outputs={
                            noded[ r + b['bus_out_suffix']] : solph.Flow(
                                variable_costs=series_variable_costs(series_var_cost[b['source_suffix']], number_timesteps),
                                emission = b['emission_factor [kton CO2/GWh]'],
                                emission_constraint = b['emission constraint'],
                                supply_2050_2055 = b['supply_2050_2055'],
                               )
                            })
                            
                nuclear=xls.parse('nuclear')
                for i, b in nuclear.iterrows():
                    for r in R:    
                            label= r + b['source_suffix']
                            noded[label] = solph.components.Source(label = label, 
                            outputs={
                            noded[r + b['bus_out_suffix']] : solph.Flow(
                                variable_costs=b['variable_costs'],
                                emission = b['emission_factor [kton CO2/GWh]'],  
                                min = b['min'],
                                investment=solph.Investment( 
                                existing= existing_capacities.loc[r,b['source_suffix']],
                                overall_maximum= overall_maximum_values.loc[r,b['source_suffix']],
                                maximum = [0,0,0,0, overall_maximum_values.loc[r,b['source_suffix']],
                                            overall_maximum_values.loc[r,b['source_suffix']], overall_maximum_values.loc[r,b['source_suffix']]],
                                ep_costs = series_ep_cost[b['source_suffix']],
                                fixed_costs = fixed_cost[b['source_suffix']].loc[0],
                                lifetime =lifetime[b['source_suffix']].loc[1],
                                age= age_values.loc[r,b['source_suffix']],
                                interest_rate =b['interest rate'],
                                cost_decrease =b['cost_decrease']) ,
                                )}
                            )
                           
                              
               
                transformers_emission = xls.parse('transformer CO2',header =0)
                for i, b in transformers_emission.iterrows():
                              for r in R:
                                           label= r + b['label']
                                           noded[label] = solph.components.Transformer(label = label, 
                                           inputs={
                                             noded[r + b['from']] : solph.Flow()},

                                           outputs={
                                               noded[r + b['bus']] : solph.Flow(),
                                               noded[r + b['bus2']] : solph.Flow(),
                                               },
                                           
                                           conversion_factors={
                                            noded[ r + b['bus']]: b['efficiency'],
                                            noded[ r + b['bus2']]: b['CO2 converted']
                                            })
                 
                emitters = xls.parse('emitters',header =0)
                for i, b in emitters.iterrows():
                               for r in R:
                                            label= r + b['label']
                                            noded[label] = solph.components.Transformer(label = label, 
                                            inputs={
                                              noded[r + b['from']] : solph.Flow()},
                                            
                                            outputs={
                                                noded[r + b['bus']] : solph.Flow(),
                                                },
                                            
                                            conversion_factors={
                                             noded[ r + b['bus']]: 1,
                                             })
                                
                            
                sinks=xls.parse('demand')
            
                for i, b in sinks.iterrows():
                              for r in R:
                                    label= r + b['demand']
                                    noded[label] = solph.components.Sink(label = label, 
                                    inputs={
                                    noded[r + b['bus_in']] : solph.Flow(
                                      nominal_value = demand_nominal_value_r.loc[r , b['demand']],
                                      emission_constraint = b['emission_constraint'],
                                      fix= timeseries[label]
                                      
                              )})          
              
                sinks=xls.parse('demand mobility')
             
                for i, b in sinks.iterrows():
                                for r in R:
                                      label= r + b['demand']
                                      noded[label] = solph.components.Sink(label = label, 
                                      inputs={
                                      noded[r + b['bus_in']] : solph.Flow(
                                        nominal_value = 1,
                                        emission_constraint = b['emision constraint'],
                                        emission_value = b['CO2 factor'],
                                        fix= timeseries[label]
                                )})
                                      
                                      
                sinks=xls.parse('demand electric vehicles')
             
                for i, b in sinks.iterrows():
                                for r in R:
                                      label= r + b['demand']
                                      noded[label] = solph.components.Sink(label = label, 
                                      inputs={
                                      noded[r + b['bus_in']] : solph.Flow(
                                        nominal_value = demand_nominal_value_r.loc[r , b['demand']],
                                        emission_constraint = b['emision constraint'],
                                        fix= timeseries[label]
                                       
                                )})                      
                                      
                sinks=xls.parse('demand_nav_avi')
             
                for i, b in sinks.iterrows():
                                for r in R:                        
                                      label= r + b['demand']
                                      noded[label] = solph.components.Sink(label = label, 
                                      inputs={
                                      noded[r + b['bus_in']] : solph.Flow(
                                        nominal_value = 1,
                                        emission_constraint = b['emision constraint'],
                                        fix= timeseries[label]
                                       
                                )})
                                      
                sinks=xls.parse('demand industry')
             
                for i, b in sinks.iterrows():
                                for r in R:
                                      label= r + '_' + b['demand']
                                      noded[label] = solph.components.Sink(label = label, 
                                      inputs={
                                      noded[r + b['bus_in']] : solph.Flow(                                 
                                        nominal_value = 1,
                                        emission_constraint = b['emision constraint'],
                                        fix= timeseries[label]                                       
                                )})
                        
                transformers = xls.parse('simple transformers')
                for i, s in transformers.iterrows():
                    for r in R:
                        if s['Fixed'] == True:
                                 
                                noded[r + s['label']] = solph.components.Transformer(
                                    label= r + s['label'],
                                    inputs={
                                        noded[r + s['from']]: solph.Flow(
                                            
                                            emission = s['CO2 emission factor [kton/GWh]']
                                              )},
                                    outputs={
                                        noded[ r + s['bus']]: solph.Flow(
                                            #nominal_value=s['capacity [MW]'],
                                            variable_costs=s['variable costs (M€/GWh)'],
                                            fix = (timeseries[s['profile (if fixed)']]),
                                            emission_constraint = s['emission constraint'],
                                            inv_ramp= s['inv_ramp'],
                                            investment=solph.Investment(                                                
                                            existing= existing_capacities.loc[r,s['label']],
                                            overall_maximum= overall_maximum_values.loc[r,s['label']],
                                            maximum = overall_maximum_values.loc[r,s['label']],#/number_periods,
                                            ep_costs = series_ep_cost[s['label']],
                                            #fixed_costs = fixed_costs_values.loc[0,s['label']],
                                            fixed_costs = fixed_cost[s['label']].loc[0],
                                            lifetime =lifetime[s['label']].loc[1],
                                            age= age_values.loc[r,b['source_suffix']],
                                            interest_rate =s['interest rate'],
                                            cost_decrease = s['cost_decrease'],
                                            ),
                                            )},
                                    conversion_factors={
                                        noded[r + s['bus']]: s['efficiency']})
                                
                              
                                
                        else:
                              noded[r + s['label']] = solph.components.Transformer(
                              label= r + s['label'],
                              inputs={
                              noded[ r + s['from']]: solph.Flow(
                                  
                                  emission = s['CO2 emission factor [Kton/GWh]'])},
                              outputs={
                              noded[ r + s['bus']]: solph.Flow(                                  
                                  max=(s['max']),
                                  min=(s['min'] ),
                                  variable_costs=s['variable costs (M€/GWh)'],
                                  emission_constraint = s['emission constraint'],
                                  inv_ramp= s['inv_ramp'],
                                  investment=solph.Investment( 
                                  existing= existing_capacities.loc[r,s['label']],
                                  overall_maximum= overall_maximum_values.loc[r,s['label']],
                                  maximum = overall_maximum_values.loc[r,s['label']],#/number_periods,
                                  ep_costs = series_ep_cost[s['label']],
                                  #fixed_costs = fixed_costs_values.loc[0,s['label']],
                                  fixed_costs = fixed_cost[s['label']].loc[0],
                                  lifetime =lifetime[s['label']].loc[1],                                  
                                  age= age_values.loc[r,s['label']],                             
                                  interest_rate =s['interest rate'],
                                  cost_decrease = s['cost_decrease'],                                  
                                  ),
                                  )},
                             conversion_factors={
                              noded[ r + s['bus']]: s['efficiency']})    
                
                transformers = xls.parse('DAC')
                for i, s in transformers.iterrows():
                    for r in R:
                        if s['Fixed'] == True:
                                 
                                noded[r + s['label']] = solph.components.Transformer(
                                    label= r + s['label'],
                                    inputs={
                                        noded[r + s['from']]: solph.Flow(
                                            
                                            emission = s['CO2 emission factor [kton/GWh]']
                                              )},
                                    outputs={
                                        noded[ r + s['bus']]: solph.Flow(
                                            variable_costs=s['variable costs (M€/GWh)'],
                                            fix = (timeseries[s['profile (if fixed)']]),
                                            emission_constraint = s['emission constraint'],
                                            investment=solph.Investment(                                                
                                            existing= existing_capacities.loc[r,s['label']],
                                            overall_maximum= overall_maximum_values.loc[r,s['label']],
                                            maximum = overall_maximum_values.loc[r,s['label']],#/number_periods,
                                            ep_costs = series_ep_cost[s['label']],
                                            #fixed_costs = fixed_costs_values.loc[0,s['label']],
                                            fixed_costs = fixed_cost[s['label']].loc[0],
                                            lifetime =lifetime[s['label']].loc[1],
                                            age= age_values.loc[r,b['source_suffix']],
                                            interest_rate =s['interest rate'],
                                            cost_decrease = s['cost_decrease'],                                            
                                            ),
                                            )},
                                    conversion_factors={
                                        noded[r + s['bus']]: s['efficiency']})
                                

                        else:
                              noded[r + s['label']] = solph.components.Transformer(
                              label= r + s['label'],
                              inputs={
                              noded[ r + s['from']]: solph.Flow(
                                  
                                  emission = s['CO2 emission factor [Kton/GWh]'])},
                              outputs={
                              noded[ r + s['bus']]: solph.Flow(
                                  max=(s['max']),
                                  min=(s['min'] ),
                                  variable_costs=s['variable costs (M€/GWh)'],
                                  emission_constraint = s['emission constraint'],
                                  investment=solph.Investment( 
                                  existing= existing_capacities.loc[r,s['label']],
                                  overall_maximum= overall_maximum_values.loc[r,s['label']],
                                  maximum= [0,overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']]],
                                  ep_costs = series_ep_cost[s['label']],                                  
                                  fixed_costs = fixed_cost[s['label']].loc[0],
                                  lifetime =lifetime[s['label']].loc[1],
                                  age= age_values.loc[r,s['label']],
                                  interest_rate =s['interest rate'],
                                  cost_decrease = s['cost_decrease'],
                                  ),
                                  )},
                             conversion_factors={
                              noded[ r + s['bus']]: s['efficiency']})       
                              
                transformers_2out = xls.parse('two output transformers')
                for i, s in transformers_2out.iterrows():
                    for r in R:
                        if s['Fixed'] == True:
                                 
                                noded[r + s['label']] = solph.components.Transformer(
                                    label= r + s['label'],
                                    inputs={
                                    noded[ r + s['from']]: solph.Flow(
                                        
                                        emission = s['CO2 emission factor [kton/GWh]'])},
                                    outputs={
                                        noded[ r + s['bus']]: solph.Flow(
                                            variable_costs=s['variable costs (M€/GWh)'],
                                            fix = (timeseries[s['profile (if fixed)']]),
                                            investment=solph.Investment( 
                                            existing= existing_capacities.loc[r,s['label']],
                                            overall_maximum= overall_maximum_values.loc[r,s['label']],
                                            maximum = overall_maximum_values.loc[r,s['label']],
                                            ep_costs = series_ep_cost[s['label']],
                                            fixed_costs = fixed_cost[s['label']].loc[0],
                                            lifetime =lifetime[s['label']].loc[1],                                            
                                            age= age_values.loc[r,s['label']],
                                            interest_rate =s['interest rate'],
                                            cost_decrease = s['cost_decrease'] 
                                            )),
                                        
                                        noded[ r + s['bus2']]: solph.Flow()                                        
                                        },
                                    conversion_factors={
                                     noded[ r + s['bus']]: s['efficiency'],
                                     noded[ r + s['bus2']]:s['CO2 converted']
                                     })
                                
                             
                        else:
                              noded[r + s['label']] = solph.components.Transformer(
                              label= r + s['label'],
                              inputs={
                              noded[ r + s['from']]: solph.Flow(                                  
                                  emission = s['CO2 emission factor [kton/GWh]'])},
                              
                              outputs={
                              noded[ r + s['bus']]: solph.Flow(
                                  max=(s['max']),
                                  min=(s['min'] ),
                                  variable_costs=s['variable costs (M€/GWh)'],
                                  investment=solph.Investment( 
                                  existing= existing_capacities.loc[r,s['label']],
                                  overall_maximum= overall_maximum_values.loc[r,s['label']],
                                  maximum = overall_maximum_values.loc[r,s['label']],#/number_periods,
                                  ep_costs = series_ep_cost[s['label']],
                                  #fixed_costs = fixed_costs_values.loc[0,s['label']],
                                  fixed_costs = fixed_cost[s['label']].loc[0],
                                  lifetime =lifetime[s['label']].loc[1],
                                  age= age_values.loc[r,s['label']],
                                  interest_rate =s['interest rate'],
                                  cost_decrease = s['cost_decrease'] 
                                  )),
                                                                      
                              noded[ r + s['bus2']]: solph.Flow()
                             
                              },
                             conversion_factors={
                              noded[ r + s['bus']]: s['efficiency'],
                              noded[ r + s['bus2']]:s['CO2 converted']
                              })
                              
                              
                
                transformers_2out = xls.parse('two output transformers- no INV')
                for i, s in transformers_2out.iterrows():
                    for r in R:
                        
                              noded[r + s['label']] = solph.components.Transformer(
                              label= r + s['label'],
                              inputs={
                              noded[ r + s['from']]: solph.Flow()                                  
                                  },
                              
                              outputs={
                              noded[ r + s['bus']]: solph.Flow(),                                                                      
                              noded[ r + s['bus2']]: solph.Flow()             
                              },
                             conversion_factors={
                              noded[ r + s['bus']]: s['efficiency'],
                              noded[ r + s['bus2']]:s['CO2 converted']
                              })              

                transformers_2out = xls.parse('two inputs transformers')
                for i, s in transformers_2out.iterrows():
                    for r in R:
                        
                              noded[r + s['label']] = solph.components.Transformer(
                              label= r + s['label'],
                              inputs={
                              noded[ r + s['from']]: solph.Flow(
                                  emission_constraint = b['emision constraint']
                                  ),
                              noded[ r + s['from2']]: solph.Flow()                             
                              
                              },
                              outputs={
                              noded[ r + s['bus']]: solph.Flow(
                                  max=(s['max']),
                                  min=(s['min'] ),
                                  variable_costs=s['variable costs (M€/GWh)'],
                                  investment=solph.Investment( 
                                  existing= existing_capacities.loc[r,s['label']],
                                  overall_maximum= overall_maximum_values.loc[r,s['label']],
                                  maximum = [0,overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']]],
                                  ep_costs = series_ep_cost[s['label']],
                                  fixed_costs = fixed_cost[s['label']].loc[0],
                                  lifetime =lifetime[s['label']].loc[1],
                                  age= age_values.loc[r,s['label']],
                                  interest_rate =s['interest rate'],                                  
                                  cost_decrease = s['cost_decrease'] 
                                  ),)},
                             conversion_factors={
                              noded[ r + s['from']]: s['efficiency'],
                              noded[ r + s['from2']]:s['el consumption']
                              }) 
                carbon_capture_tech = xls.parse('Carbon Capture')
                for i, s in carbon_capture_tech.iterrows():
                    for r in R:
                        
                              noded[r + s['label']] = solph.components.Transformer(
                              label= r + s['label'],
                              inputs={
                              noded[ r + s['from']]: solph.Flow(
                                  emission_constraint = b['emision constraint']
                                  ),
                              noded[ r + s['from2']]: solph.Flow()                              
                              
                              },
                              outputs={
                              noded[ r + s['bus']]: solph.Flow(
                                  max=(s['max']),
                                  min=(s['min'] ),
                                  variable_costs=s['variable costs (M€/GWh)'],
                                  investment=solph.Investment( 
                                  existing= existing_capacities.loc[r,s['label']],
                                  overall_maximum= overall_maximum_values.loc[r,s['label']],
                                  maximum = [0,overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']]],
                                  ep_costs = series_ep_cost[s['label']],
                                  fixed_costs = fixed_cost[s['label']].loc[0],
                                  lifetime =lifetime[s['label']].loc[1],
                                  age= age_values.loc[r,s['label']], 
                                  interest_rate =s['interest rate'],                                  
                                  cost_decrease = s['cost_decrease'] 
                                  ),
                                  ),
                              
                              noded[ r + s['bus2']]: solph.Flow()
                              
                              },
                             conversion_factors={
                              noded[ r + s['from']]: s['efficiency'],
                              noded[ r + s['from2']]:s['el consumption'],
                              noded[ r + s['bus2']]:1-s['Capture efficciency'],
                              noded[ r + s['bus']]:s['Capture efficciency'],
                              })
                
                ###### new electrical grid
                new_grid_el = xls.parse('new_elec_line_2030',header =0, index_col=0)
                distance_region_EE = xls.parse('dist regions EE',header =0, index_col=0)
                for r in R: 
                  for re in R:  
                      if new_grid_el.loc [r , re]  > 0 :
                                            label= r + '_new_line_ele_2030_' + re
                                            noded[label] = solph.components.Transformer(label = label, 
                                            inputs={
                                              noded[r + '_bus_el'] : solph.Flow(                                                 
                                                  investment=solph.Investment( 
                                                  existing= 0,
                                                  overall_maximum= new_grid_el.loc[r, re],
                                                  maximum = [0,new_grid_el.loc[r, re],new_grid_el.loc[r, re], new_grid_el.loc[r, re],new_grid_el.loc[r, re],new_grid_el.loc[r, re],new_grid_el.loc[r, re]],
                                                  ep_costs = 0 ,
                                                  fixed_costs = 0 ,
                                                  lifetime = 50,
                                                  interest_rate =0.06,
                                                  )
                                              )},

                                            outputs={
                                                noded[re + '_bus_el'] : solph.Flow(
                                                )},
                                           
                                            conversion_factors={
                                            noded[ re + '_bus_el']: 1 - 0.07 * distance_region_EE.loc[r, re] /1000})
                
                new_grid_el_2035 = xls.parse('new_elec_line_2035',header =0, index_col=0)
                for r in R: 
                  for re in R:  
                    if new_grid_el_2035.loc [r , re]  > 0 :
                                            label= r + '_new_line_ele_2035_' + re
                                            noded[label] = solph.components.Transformer(label = label, 
                                            inputs={
                                              noded[r + '_bus_el'] : solph.Flow(
                                                  investment=solph.Investment( 
                                                  existing= 0,
                                                  overall_maximum= new_grid_el_2035.loc[r, re],
                                                  maximum = [0,0,new_grid_el_2035.loc[r, re], new_grid_el_2035.loc[r, re],new_grid_el_2035.loc[r, re],new_grid_el_2035.loc[r, re],new_grid_el_2035.loc[r, re]],
                                                  ep_costs = 0 ,
                                                  fixed_costs = 0 ,
                                                  lifetime = 50,
                                                  interest_rate =0.06,
                                                  )
                                              )},

                                            outputs={
                                                noded[re + '_bus_el'] : solph.Flow(
                                                )},
                                           
                                            conversion_factors={
                                            noded[ re + '_bus_el']: 1 - 0.07 * distance_region_EE.loc[r, re] /1000})
                                            
                new_grid_el_2040 = xls.parse('new_elec_line_2040',header =0, index_col=0)
                for r in R: 
                  for re in R:                  
                    if new_grid_el_2040.loc [r , re]  > 0 :
                                            label= r + '_new_line_ele_2040_' + re
                                            noded[label] = solph.components.Transformer(label = label, 
                                            inputs={
                                              noded[r + '_bus_el'] : solph.Flow(
                                                  investment=solph.Investment( 
                                                  existing= 0,
                                                  overall_maximum= new_grid_el_2040.loc[r, re],
                                                  maximum = [0,0,0, new_grid_el_2040.loc[r, re],new_grid_el_2040.loc[r, re],new_grid_el_2040.loc[r, re],new_grid_el_2040.loc[r, re]],
                                                  ep_costs = 0 ,                                                  
                                                  fixed_costs = 0 ,
                                                  lifetime = 50,
                                                  interest_rate =0.06,
                                                  )
                                              )},

                                            outputs={
                                                noded[re + '_bus_el'] : solph.Flow(
                                                )},
                                            conversion_factors={
                                            noded[ re + '_bus_el']: 1 - 0.07 * distance_region_EE.loc[r, re] /1000})                           
             
                
                ####### Collegamenti Rete elettrica #######
                line_matrix = xls.parse('line_matrix',header =0, index_col=0)
                
                for r in R: 
                  for re in R:                    
                    if line_matrix.loc [r , re]  > 0 :
                                            label= r + '_line_' + re
                                            noded[label] = solph.components.Transformer(label = label, 
                                            inputs={
                                              noded[r + '_bus_el'] : solph.Flow(
                                                  link_par = line_matrix_constraint.loc[r,re],
                                                  nominal_value= line_matrix.loc [r , re]
                                              )},

                                            outputs={
                                                noded[re + '_bus_el'] : solph.Flow(
                                                )},
                                            
                                            conversion_factors={
                                             noded[ re + '_bus_el']: 1 - 0.07 * distance_region_EE.loc[r, re] /1000 })
               
                ####### Nuovi Collegamenti Rete elettrica #######
                if ee_grid_invest_case== True:
                    line_matrix = xls.parse('new_elec_inv_line',header =0, index_col=0)
                    distance_region_EE = xls.parse('dist regions EE',header =0, index_col=0)
                    for r in R: 
                      for re in R:                                            
                        if distance_region_EE.loc [r , re]  > 0 :
                                                label= r + '_new_line_ele_inv_' + re
                                                noded[label] = solph.components.Transformer(label = label, 
                                                inputs={
                                                  noded[r + '_bus_el'] : solph.Flow(
                                                      investment=solph.Investment( 
                                                      existing= 0,
                                                      overall_maximum= 50,
                                                      maximum = [0,0,0,50,50,50,50],
                                                      ep_costs = 0.433* distance_region_EE.loc[r, re]  ,                                                      
                                                      fixed_costs = 0,                                                           
                                                      lifetime = 40,
                                                      interest_rate =0.06,
                                                      )
                                                  )},

                                                outputs={
                                                    noded[re + '_bus_el'] : solph.Flow(
                                                    )},
                                               
                                                conversion_factors={
                                                noded[ re + '_bus_el']: 1 - 0.07 * distance_region_EE.loc[r, re] /1000})
                        
 
                ##### Collegamenti Rete Gas ########
                link_blending_matrix = xls.parse('link_blending_matrix',header =0, index_col=0)
                link_blend = xls.parse('link')
                for i, s in link_blend.iterrows():
                    for r in R: 
                      for re in R:                              
                          if link_blending_matrix.loc [r , re]  > 0 :
                                                label= r + '_gas_blending_line_' + re
                                                noded[label] = solph.components.experimental.Link(label = label, 
                                                inputs={
                                                  noded[r + '_bus_gas'] : solph.Flow(
                                                      variable_costs = 0.0027,
                                                  ),
                                                  noded[r + '_bus_h2_blending'] : solph.Flow(     
                                                      variable_costs = 0.0027,
                                                  )
                                                  },

                                                outputs={
                                                    noded[re + '_bus_gas'] : solph.Flow(),
                                                    noded[re + '_bus_h2_blending'] : solph.Flow()
                                                    },
                                                
                                                conversion_factors={
                                                  (noded[ r + '_bus_gas'],noded[ re + '_bus_gas']): s['conversion_factor_1'],
                                                  (noded[ r + '_bus_h2_blending'],noded[ re + '_bus_h2_blending']): s['conversion_factor_2'],

                                                  },
                                                area_limit= link_blending_matrix.loc [r , re],
                                                max_percentage_h2= blend_max_percentage_case
                                                )

                transformers_blending = xls.parse('blending_trasformer')
                for i, s in transformers_blending.iterrows():
                   for r in R:
                       noded[r + s['label']] = solph.components.experimental.Blending(
                       label= r + s['label'],
                       inputs={
                       noded[ r + s['from']]: solph.Flow(
                           emission_constraint = s['emission_constraint']                           
                           ),
                       noded[ r + s['from2']]: solph.Flow()
                       },
                       outputs={
                       noded[ r + s['bus']]: solph.Flow()},
                       max_percentage_h2 =  blend_max_percentage_case
                      )
                                                    
                #### rete idrogeno ####
                if hydrogen_invest_case== True:
                    h2_links_matrix = xls.parse('h2_link_matrix',header =0, index_col=0)
                    distance_region = xls.parse('distance regions',header =0, index_col=0)
                    for r in R: 
                      for re in R:    
                        if h2_links_matrix.loc [r , re]  > 0 :
                                                label= r + '_line_h2_' + re
                                                noded[label] = solph.components.Transformer(label = label, 
                                                inputs={
                                                  noded[r + '_bus_h2'] : solph.Flow(
                                                        investment=solph.Investment( 
                                                        existing= 0,
                                                        overall_maximum= h2_links_matrix.loc[r, re]*3,
                                                        maximum = [0,0,h2_links_matrix.loc[r, re]*3, h2_links_matrix.loc[r, re]*3,h2_links_matrix.loc[r, re]*3,h2_links_matrix.loc[r, re]*3,h2_links_matrix.loc[r, re]*3],
                                                        ep_costs = 0.226 * distance_region.loc[r, re],
                                                        fixed_costs =  0,
                                                        age= 0,     
                                                        lifetime = 50,
                                                        interest_rate =0.06,
                                                        )
                                                  )},

                                                outputs={
                                                    noded[re + '_bus_h2'] : solph.Flow(
                                                    )},
                                               
                                                conversion_factors={
                                                noded[ re + '_bus_h2']: 0.99})                             
                                                
                #### rete CO2 ####
                if co2_grid_invest_case == True:
                    distance_region = xls.parse('distance regions',header =0, index_col=0)
                    link_co2_matrix = xls.parse('co2_links',header =0, index_col=0)
                    for r in R: 
                      for re in R:
                        if link_co2_matrix.loc [r , re]  > 0 :
                                                label= r + '_line_co2_' + re
                                                noded[label] = solph.components.Transformer(label = label, 
                                                inputs={
                                                  noded[r + '_bus_CO2_sequestred'] : solph.Flow(
                                                     
                                                        investment=solph.Investment( 
                                                        existing= 0,
                                                        overall_maximum= link_co2_matrix.loc[r, re]*1000,
                                                        maximum = [0,link_co2_matrix.loc[r, re]*1000,link_co2_matrix.loc[r, re]*1000, link_co2_matrix.loc[r, re]*1000,link_co2_matrix.loc[r, re]*1000,link_co2_matrix.loc[r, re]*1000,link_co2_matrix.loc[r, re]*1000],
                                                        ep_costs = 2.2 * distance_region.loc[r, re],
                                                        fixed_costs =  0,
                                                        age= 0,    
                                                        lifetime = 50,
                                                        interest_rate =0.06, 
                                                        )
                                                  )},

                                                outputs={
                                                    noded[re + '_bus_CO2_sequestred'] : solph.Flow(
                                                    )},
                                               
                                                conversion_factors={
                                                noded[ re + '_bus_CO2_sequestred']: 1})
                
                #### rete LF ####
                if LF_grid_case == True:
                    link_LF_matrix = xls.parse('LF_links',header =0, index_col=0)
                    for r in R: 
                      for re in R:  
                        if link_LF_matrix.loc [r , re]  > 0 :
                                                label= r + '_line_LF_' + re
                                                noded[label] = solph.components.Transformer(label = label, 
                                                inputs={
                                                  noded[r + '_bus_LF'] : solph.Flow(                                                     
                                                        variable_costs= 0.0022,
                                                  )},

                                                outputs={
                                                    noded[re + '_bus_LF'] : solph.Flow(
                                                    )},
                                               
                                                conversion_factors={
                                                noded[ re + '_bus_LF']: 1})
                
                                           
                storage_char_cap = xls.parse('storage char power capacity',header =0, index_col=0)
                storage_dischar_cap = xls.parse('storage dis power capacity',header =0, index_col=0)
                storage_ene_cap = xls.parse('storage energy capacity',header =0, index_col=0)
                storage_initial_cap = xls.parse('storage initial capacity',header =0, index_col=0)
                storages = xls.parse('storages')
                for i, s in storages.iterrows():
                    for r in R:
                                  if s['label'] == '_hydro_pump':
                                      noded[ r + s['label']] = solph.components.GenericStorage(
                                      label= r + s['label'],
                                      inputs={
                                          noded[ r + s['bus 1 in']]: solph.Flow(
                                                variable_costs=s['variable costs (M€/GWh)'],
                                                nominal_value = storage_char_cap.loc[r , s['label']],
                                              ),
                                          },
                                      outputs={ 
                                          noded[ r + s['bus out']]: solph.Flow(
                                                nominal_value = storage_dischar_cap.loc[r , s['label']],
                                              )},
                                      balanced = True,
                                      loss_rate=s['capacity loss'],
                                      nominal_storage_capacity = storage_ene_cap.loc[r , s['label']],
                                      initial_storage_level=storage_initial_cap.loc[r , s['label']],
                                      max_storage_level=s['cmax'],
                                      min_storage_level=s['cmin'],
                                      inflow_conversion_factor=s['charge efficiency'],
                                      outflow_conversion_factor=s['discharge efficiency']
                                      )
                                  elif s['label'] == '_hydro_res' :
                                      noded[ r + s['label']] = solph.components.GenericStorage(
                                      label= r + s['label'],
                                      inputs={
                                          noded[ r + s['bus 1 in']]: solph.Flow(
                                                variable_costs=s['variable costs (M€/GWh)'],
                                                ),                                        
                                          
                                          },
                                      outputs={ 
                                          noded[ r + s['bus out']]: solph.Flow(
                                                nominal_value = storage_dischar_cap.loc[r , s['label']],
                                              )},
                                      balanced = True,
                                      loss_rate=s['capacity loss'],
                                      nominal_storage_capacity = storage_ene_cap.loc[r , s['label']],
                                      initial_storage_level=storage_initial_cap.loc[r , s['label']],
                                      max_storage_level=s['cmax'],
                                      min_storage_level=s['cmin'],
                                      inflow_conversion_factor=s['charge efficiency'],
                                      outflow_conversion_factor=s['discharge efficiency']
                                      )
                                  elif s['label'] == '_waste_stor':
                                      noded[ r + s['label']] = solph.components.GenericStorage(
                                      label= r + s['label'],
                                      inputs={
                                          noded[ r + s['bus 1 in']]: solph.Flow(
                                                variable_costs=s['variable costs (M€/GWh)'],
                                                nominal_value = storage_char_cap.loc[r , s['label']],
                                              ),                                 
                                          
                                          },
                                      outputs={ 
                                          noded[ r + s['bus out']]: solph.Flow(
                                                nominal_value = storage_dischar_cap.loc[r , s['label']],
                                              )},
                                      balanced = True,
                                      loss_rate=s['capacity loss'],
                                      nominal_storage_capacity = storage_ene_cap.loc[r , s['label']],
                                      initial_storage_level=0.5,
                                      max_storage_level=s['cmax'],
                                      min_storage_level=s['cmin'],
                                      inflow_conversion_factor=s['charge efficiency'],
                                      outflow_conversion_factor=s['discharge efficiency']
                                      )    
                                      
                                  else:
                                      noded[ r + s['label']] = solph.components.GenericStorage(
                                      label= r + s['label'],
                                      investment = solph.Investment( 
                                      existing= existing_capacities.loc[r,s['label']],
                                      maximum= [0,overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']],overall_maximum_values.loc[r,s['label']]],
                                      ep_costs = series_ep_cost[s['label']],
                                      fixed_costs = fixed_cost[s['label']].loc[0],
                                      lifetime =lifetime[s['label']].loc[1],
                                      age= age_values.loc[r,s['label']], 
                                      interest_rate =s['interest rate'],                                      
                                      cost_decrease = 0.1
                                      ),
                                      invest_relation_input_capacity = s['invest_relation_input_capacity'],
                                      invest_relation_output_capacity = s['invest_relation_input_capacity'],
                                      lifetime_inflow = lifetime[s['label']].loc[1],
                                      lifetime_outflow = lifetime[s['label']].loc[1],
                                      inputs={
                                          noded[ r + s['bus 1 in']]: solph.Flow(
                                                variable_costs=s['variable costs (M€/GWh)']
                                              )},
                                      outputs={
                                          noded[ r + s['bus out']]: solph.Flow(
                                              )},
                                      balanced = True,
                                      loss_rate=s['capacity loss'],
                                      initial_storage_level=1,
                                      max_storage_level=s['cmax'],
                                      min_storage_level=s['cmin'],
                                      inflow_conversion_factor=s['charge efficiency'],
                                      outflow_conversion_factor=s['discharge efficiency']
                                      )
                                  
                return noded

    #%%
                      
    number_timesteps   = 24*clustering_options['n_clusters']
    logger.define_logging()    
    datetime_index_dict={}    
    for year in range(2025, 2056, delta):
        datetime_index_dict[year]= pd.date_range(f'1/1/{year}', periods = number_timesteps, freq='H')

    periodi ={}
    i=0
    for key,value in datetime_index_dict.items():
        periodi[i]= datetime_index_dict[key]
        i= i+1
    
    flattened_list = []
    if number_typ_days != 365:
        for sublist in repetition_per_hour:
            for item in sublist:
                flattened_list.append(item)
    
        repetition_per_hour_model =  flattened_list
        
        flattened_list1 = []
        year_flatten=-1
        final_list_model =[]
        for sublist in final_list1:
            year_flatten= year_flatten +1
            for item in final_list1[sublist]:
                item= item + 24*number_typ_days*year_flatten
                flattened_list1.append(item)
            
        final_list_model.extend(flattened_list1)        
    
        s =0
        for key in range(number_periods):
            s = s + 1
        #### ripetizione sui periodi  di first index #####
        first_index1 = []
        for i in range(s):
            new_list = [j + (365*i) for j in primi_giorni_clusters[i]]
            first_index1 += new_list

        # Define the start date and first_index
        start_date = '2018-01-01'

        # Calculate the date for each value in first_index
        dates = [pd.Timestamp(start_date) + pd.DateOffset(days=i) for i in first_index1]

        # Format the dates as strings in the desired format
        dates = [d.strftime('%Y-%m-%d') for d in dates]

        # Define the start and end dates for the date range
        start_date = '2015-01-01 00:00:00'
        end_date = '2060-12-31 23:59:59'

        # Define the 8 non-consecutive days you want to include in the date range
        #days = [pd.to_datetime(d) for d in ['2022-01-01', '2022-02-01', '2022-03-01','2022-04-01', '2022-05-01','2022-06-01', '2022-07-01','2022-08-01','2022-09-01','2022-10-01','2022-11-01','2022-12-01']]
        days = [pd.to_datetime(d) for d in dates]
        hours = [pd.to_datetime(f"{d.date()} {h:02d}:00:00") for d in days for h in range(24)]
        last_timestamp = hours[-1]
        next_timestamp = last_timestamp + pd.Timedelta(hours=1)
        hours.append(next_timestamp)
        days_hours = days + hours
        date_range = pd.date_range(start_date, end_date, freq='H')
        date_range = date_range[date_range.isin(days_hours)]
        datetime_index_all = date_range    
        
    ###############################################################
    # %%

    es = solph.EnergySystem(timeindex=datetime_index_all, periods = periodi , multi_period=True,infer_last_interval=False, clustering_matrix = final_list_model, delta_year_periods = delta_year_periods, repetition_per_hour = repetition_per_hour_model)
             
    nodes = nodes_from_excel(
        os.path.join(os.path.dirname(__file__), InputData,),profiles)

    es.add(*nodes.values())
                
    print("********************************************************")
    print("The following objects has been created from excel sheet:")
    for n in es.nodes:
                    oobj = str(type(n)).replace("<class 'oemof.solph.", "").replace("'>", "")
                    print(oobj + ':', n.label)
                    
                    
    om = solph.Model(es,discount_rate = discount_rate)
    
    line_matrix_constraint = xls.parse('line_matrix_con', header=0, index_col=0)
    line_matrix = xls.parse('line_matrix', header=0, index_col=0)

    #%% Vincolo emissioni nulle ultimi due periodi
    import pyomo.environ as po

    ####################################
    def emission_con (om):
        
        flows_CO2_emitted ={}
        flows_CO2_captured = {}
        flows_CO2_sink_LF = {}
        flows_CO2_sink_gas = {}
        flows_CO2_bio_fuel = {}
        flows_CO2_biomass_sustinable = {}
        flows_CO2_biomass_emitted = {}        
        flows_CO2_biomethane_not_accounted = {}
        flows_CO2_import_el = {}
        flows_CO2_import_h2_blue ={}
        flows_CO2_emitted_CC ={}
        
        for (i, o) in om.flows:
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 1:
                        flows_CO2_captured[(i, o)] = om.flows[i, o]                    
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 2:
                        flows_CO2_emitted[(i, o)] = om.flows[i, o]                        
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 4:
                        flows_CO2_sink_LF[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 3:
                        flows_CO2_sink_gas[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 6:
                        flows_CO2_bio_fuel[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 7:
                        flows_CO2_biomass_sustinable[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 8:
                        flows_CO2_biomass_emitted[(i, o)] = om.flows[i, o]                    
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 9:
                        flows_CO2_biomethane_not_accounted[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 10:
                        flows_CO2_import_el[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 11:
                        flows_CO2_import_h2_blue[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 12:
                        flows_CO2_emitted_CC[(i, o)] = om.flows[i, o]
                        
                       
        periodi_CO2 = sorted(list(set(om.es.periods.keys())))
        om.PERIODS_CO2 = po.Set(
                initialize= periodi_CO2[-2:]
            )   ##### to change for flexibility #####
                      
        def rule_emission_con(om, pp):
            lhs1 = 0.0
            rhs1 = 0.0
            TIMEINDEX_MOD = [(p, t) for (p, t) in om.TIMEINDEX if p == pp]
            for p, t in TIMEINDEX_MOD:
                lhs = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_CO2_emitted)
                rhs = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_CO2_captured)
                lhs_sink_LF = sum(om.flow[i, o, p, t] * 0.291 *om.repetition_per_hour[t] for (i, o) in flows_CO2_sink_LF)
                lhs_sink_gas = sum(om.flow[i, o, p, t] * 0.202 *om.repetition_per_hour[t] for (i, o) in flows_CO2_sink_gas)
                rhs_bio_fuel = sum(om.flow[i, o, p, t] * 0.291 *om.repetition_per_hour[t] for (i, o) in flows_CO2_bio_fuel)                
                rhs_biomass_sust = sum(om.flow[i, o, p, t] * 0.41 *om.repetition_per_hour[t] for (i, o) in flows_CO2_biomass_sustinable)
                lhs_biomass_emitt = sum(om.flow[i, o, p, t] * 0.41 *om.repetition_per_hour[t] for (i, o) in flows_CO2_biomass_emitted)                
                rhs_CO2_biomethane_not_accounted = sum(om.flow[i, o, p, t] * 0.202 *om.repetition_per_hour[t] for (i, o) in flows_CO2_biomethane_not_accounted)
                lhs_CO2_import_h2_blue = sum(om.flow[i, o, p, t] * 0.09 *om.repetition_per_hour[t] for (i, o) in flows_CO2_import_h2_blue)
                lhs_CO2_import_el = sum(om.flow[i, o, p, t] * 0.1 *om.repetition_per_hour[t] for (i, o) in flows_CO2_import_el)
                lhs_CO2_emitted_cc = sum(om.flow[i, o, p, t] * 0.0 *om.repetition_per_hour[t] for (i, o) in flows_CO2_emitted_CC)
                lhs1 += lhs + lhs_sink_LF + lhs_sink_gas + lhs_biomass_emitt  + lhs_CO2_import_el  + lhs_CO2_import_h2_blue + lhs_CO2_emitted_cc
                rhs1 += rhs  + rhs_biomass_sust + rhs_CO2_biomethane_not_accounted + rhs_bio_fuel
                
            lhs_LULUCF_unavoidable = 7100
            expr= lhs1 - rhs1 + lhs_LULUCF_unavoidable   
            
            return expr == 0
        
        om.emission_const = po.Constraint(om.PERIODS_CO2 , rule=rule_emission_con)        
        return om

    om = emission_con(om)    


    ###emission 2030    ####################################
    def emission_con_2030 (om):
        
        flows_CO2_emitted ={}
        flows_CO2_captured = {}
        flows_CO2_sink_LF = {}
        flows_CO2_sink_gas = {}
        flows_CO2_bio_fuel = {}
        flows_CO2_biomass_sustinable = {}
        flows_CO2_biomass_emitted = {}
        flows_CO2_import_h2_blue ={}
        flows_CO2_biomethane_not_accounted = {}
        flows_CO2_import_el = {}
        flows_CO2_emitted_CC ={}        
        
        for (i, o) in om.flows:
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 1:
                        flows_CO2_captured[(i, o)] = om.flows[i, o]                    
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 2:
                        flows_CO2_emitted[(i, o)] = om.flows[i, o]                        
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 4:
                        flows_CO2_sink_LF[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 3:
                        flows_CO2_sink_gas[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 6:
                        flows_CO2_bio_fuel[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 7:
                        flows_CO2_biomass_sustinable[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 8:
                        flows_CO2_biomass_emitted[(i, o)] = om.flows[i, o]                    
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 9:
                        flows_CO2_biomethane_not_accounted[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 10:
                        flows_CO2_import_el[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 11:
                        flows_CO2_import_h2_blue[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 12:
                        flows_CO2_emitted_CC[(i, o)] = om.flows[i, o]
                         
                        
        def rule_emission_con_2030(om):
            lhs1 = 0.0
            rhs1 = 0.0
            TIMEINDEX_MOD = [(p, t) for (p, t) in om.TIMEINDEX if p == 1] ### to change for flexibility ####
            for p, t in TIMEINDEX_MOD:
                lhs = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_CO2_emitted)
                rhs = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_CO2_captured)
                lhs_sink_LF = sum(om.flow[i, o, p, t] * 0.291 *om.repetition_per_hour[t] for (i, o) in flows_CO2_sink_LF)
                lhs_sink_gas = sum(om.flow[i, o, p, t] * 0.202 *om.repetition_per_hour[t] for (i, o) in flows_CO2_sink_gas)
                rhs_bio_fuel = sum(om.flow[i, o, p, t] * 0.291 *om.repetition_per_hour[t] for (i, o) in flows_CO2_bio_fuel)                
                rhs_biomass_sust = sum(om.flow[i, o, p, t] * 0.41 *om.repetition_per_hour[t] for (i, o) in flows_CO2_biomass_sustinable)
                lhs_biomass_emitt = sum(om.flow[i, o, p, t] * 0.41 *om.repetition_per_hour[t] for (i, o) in flows_CO2_biomass_emitted)                
                rhs_CO2_biomethane_not_accounted = sum(om.flow[i, o, p, t] * 0.202 *om.repetition_per_hour[t] for (i, o) in flows_CO2_biomethane_not_accounted)
                lhs_CO2_import_el = sum(om.flow[i, o, p, t] * 0.1 *om.repetition_per_hour[t] for (i, o) in flows_CO2_import_el)
                lhs_CO2_import_h2_blue = sum(om.flow[i, o, p, t] * 0.09 *om.repetition_per_hour[t] for (i, o) in flows_CO2_import_h2_blue)
                lhs_CO2_emitted_cc = sum(om.flow[i, o, p, t] * 0.0 *om.repetition_per_hour[t] for (i, o) in flows_CO2_emitted_CC)
                lhs1 += lhs + lhs_sink_LF + lhs_sink_gas + lhs_biomass_emitt  + lhs_CO2_import_el + lhs_CO2_import_h2_blue + lhs_CO2_emitted_cc
                rhs1 += rhs  + rhs_biomass_sust + rhs_CO2_biomethane_not_accounted + rhs_bio_fuel #+ lhs_LULUCF_unavoidable
            
            lhs_LULUCF_unavoidable = 20400
            expr= lhs1 - rhs1 + lhs_LULUCF_unavoidable   
            
            return expr <= 233400
        
        om.emission_const_2030 = po.Constraint( rule=rule_emission_con_2030)        
        return om

    om = emission_con_2030(om)

    ################ vincolo emissioni comulate  ####################
    def emission_con_comulated (om,comulative_case_1):
        
        flows_CO2_emitted ={}
        flows_CO2_captured = {}
        flows_CO2_sink_LF = {}
        flows_CO2_sink_gas = {}
        flows_CO2_bio_fuel = {}
        flows_CO2_biomass_sustinable = {}
        flows_CO2_biomass_emitted = {}        
        flows_CO2_biomethane_not_accounted = {}
        flows_CO2_import_el = {}
        flows_CO2_import_h2_blue={}
        flows_CO2_emitted_CC ={}
        
        #p= len(om.es.periods)-1
        for (i, o) in om.flows:
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 1:
                        flows_CO2_captured[(i, o)] = om.flows[i, o]                    
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 2:
                        flows_CO2_emitted[(i, o)] = om.flows[i, o]                        
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 4:
                        flows_CO2_sink_LF[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 3:
                        flows_CO2_sink_gas[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 6:
                        flows_CO2_bio_fuel[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 7:
                        flows_CO2_biomass_sustinable[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 8:
                        flows_CO2_biomass_emitted[(i, o)] = om.flows[i, o]                    
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 9:
                        flows_CO2_biomethane_not_accounted[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 10:
                        flows_CO2_import_el[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 11:
                        flows_CO2_import_h2_blue[(i, o)] = om.flows[i, o]
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 12:
                        flows_CO2_emitted_CC[(i, o)] = om.flows[i, o]
                        
                        
                       
        periodi_CO2 = sorted(list(set(om.es.periods.keys())))
        om.PERIODS_CO2_path = po.Set(
                initialize= periodi_CO2[:-1]
            )

        unavoidable_list = [60000,54000,49500,46200,43700,43200,43200]
        LULUFC = [-36100,-36100,-36100,-36100,-36100,-36100,-36100]
                  
        def rule_emission_con_comulated(om):
            
            comulated_value = 0
            target_value = comulative_case_1 #1220000#
            for pp in om.PERIODS_CO2_path:
                TIMEINDEX_MOD = [(p, t) for (p, t) in om.TIMEINDEX if p == pp]
                lhs1 = 0.0
                rhs1 = 0.0

                for p, t in TIMEINDEX_MOD:
                    lhs = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_CO2_emitted)
                    rhs = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_CO2_captured)
                    lhs_sink_LF = sum(om.flow[i, o, p, t] * 0.291 *om.repetition_per_hour[t] for (i, o) in flows_CO2_sink_LF)
                    lhs_sink_gas = sum(om.flow[i, o, p, t] * 0.202 *om.repetition_per_hour[t] for (i, o) in flows_CO2_sink_gas)
                    rhs_bio_fuel = sum(om.flow[i, o, p, t] * 0.291 *om.repetition_per_hour[t] for (i, o) in flows_CO2_bio_fuel)                    
                    rhs_biomass_sust = sum(om.flow[i, o, p, t] * 0.41 *om.repetition_per_hour[t] for (i, o) in flows_CO2_biomass_sustinable)
                    lhs_biomass_emitt = sum(om.flow[i, o, p, t] * 0.41 *om.repetition_per_hour[t] for (i, o) in flows_CO2_biomass_emitted)                    
                    rhs_CO2_biomethane_not_accounted = sum(om.flow[i, o, p, t] * 0.202 *om.repetition_per_hour[t] for (i, o) in flows_CO2_biomethane_not_accounted)
                    lhs_CO2_import_h2_blue = sum(om.flow[i, o, p, t] * 0.09 *om.repetition_per_hour[t] for (i, o) in flows_CO2_import_h2_blue)
                    lhs_CO2_import_el = sum(om.flow[i, o, p, t] * 0.1 *om.repetition_per_hour[t] for (i, o) in flows_CO2_import_el)
                    lhs_CO2_emitted_cc = sum(om.flow[i, o, p, t] * 0.0 *om.repetition_per_hour[t] for (i, o) in flows_CO2_emitted_CC)
                    lhs1 += lhs + lhs_sink_LF + lhs_sink_gas + lhs_biomass_emitt  + lhs_CO2_import_el + lhs_CO2_import_h2_blue + lhs_CO2_emitted_cc
                    rhs1 += rhs  + rhs_biomass_sust + rhs_CO2_biomethane_not_accounted + rhs_bio_fuel
                
                lhs_LULUCF_unavoidable = unavoidable_list[pp] + LULUFC[pp]  
                expr= lhs1 - rhs1 + lhs_LULUCF_unavoidable
                comulated_value = comulated_value + expr *delta_year_periods
           
            return comulated_value <= target_value
        
        om.emission_const_comulated = po.Constraint( rule=rule_emission_con_comulated)
        
        return om

    om = emission_con_comulated(om,comulative_case)

    #%% Vincolo coneesioni BZ elettriche

    def w_tot(om, flows=None):
        if flows is None:
            flows = {}
            flows_N_CN = {}            
            flows_CN_N = {}
            flows_CN_CS = {}
            flows_CS_CN = {}
            flows_CS_S = {}
            flows_S_CS = {}
            
            for (i, o) in om.flows:
                if hasattr(om.flows[i, o], 'link_par') and om.flows[i,o].link_par == 1:
                    flows_N_CN[(i, o)] = om.flows[i, o]                
                if hasattr(om.flows[i, o], 'link_par') and om.flows[i,o].link_par == 2:
                    flows_CN_N[(i, o)] = om.flows[i, o]
                if hasattr(om.flows[i, o], 'link_par') and om.flows[i,o].link_par == 3:
                    flows_CN_CS[(i, o)] = om.flows[i, o]
                if hasattr(om.flows[i, o], 'link_par') and om.flows[i,o].link_par == 4:
                    flows_CS_CN[(i, o)] = om.flows[i, o]
                if hasattr(om.flows[i, o], 'link_par') and om.flows[i,o].link_par == 5:
                    flows_CS_S[(i, o)] = om.flows[i, o]
                if hasattr(om.flows[i, o], 'link_par') and om.flows[i,o].link_par == 6:
                    flows_S_CS[(i, o)] = om.flows[i, o]
                    
                    
        else:
            for (i, o) in om.flows:
                if not hasattr(flows[i, o], 'link_par'):
                    raise ValueError(f'Flow with source: {i.label} and target: {o.label} has no attribute w')

        dict_flow = {}
        for (i, o) in flows:
            if hasattr(flows[i, o], 'link_par'):
                if flows[i, o].link_par in dict_flow:
                    dict_flow[flows[i, o].link_par].append(flows[i, o])
                else:
                    dict_flow[flows[i, o].link_par] = [flows[i, o]]

        def link_rule_N_CN(om, p, t):
            
            lhs = sum(om.flow[i, o, p, t] for (i, o) in flows_N_CN)
            rhs= 4.3        
            return lhs <= rhs

        om.link_N_CN = po.Constraint(om.TIMEINDEX, rule=link_rule_N_CN)
        
        def link_rule_CN_N(om, p, t):
            
            lhs = sum(om.flow[i, o, p, t] for (i, o) in flows_CN_N)      
            rhs= 3.1       
            return lhs <= rhs

        om.link_CN_N = po.Constraint(om.TIMEINDEX, rule=link_rule_CN_N)
        
        def link_rule_CN_CS(om, p, t):
            
            lhs = sum(om.flow[i, o, p, t] for (i, o) in flows_CN_CS)
            rhs= 2.9
            return lhs <= rhs

        om.link_CN_CS = po.Constraint(om.TIMEINDEX, rule=link_rule_CN_CS)
        def link_rule_CS_CN(om, p, t):
            
            lhs = sum(om.flow[i, o, p, t] for (i, o) in flows_CS_CN)
            rhs= 2.7
            return lhs <= rhs

        om.link_CS_CN = po.Constraint(om.TIMEINDEX, rule=link_rule_CS_CN)
        def link_rule_CS_S(om, p, t):
            
            lhs = sum(om.flow[i, o, p, t] for (i, o) in flows_CS_S)
            rhs= 2
            return lhs <= rhs

        om.link_CS_S = po.Constraint(om.TIMEINDEX, rule=link_rule_CS_S)
        def link_rule_S_CS(om, p, t):
            
            lhs = sum(om.flow[i, o, p, t] for (i, o) in flows_S_CS)
            rhs= 5        
            return lhs <= rhs

        om.link_S_CS = po.Constraint(om.TIMEINDEX, rule=link_rule_S_CS)      

        
        return om

    om  = w_tot(om)

    #%% Vincolo gas h2 import

    def gas_h2_supply(om, flows=None):
        if flows is None:
            flows = {}
            flows_gas_1 = {}            
            flows_gas_6 = {}
            flows_gas_16 = {}
            flows_gas_19 = {}
            flows_h2_1 = {}            
            flows_h2_6 = {}
            flows_h2_16 = {}
            flows_h2_19 = {}
        
            for (i, o) in om.flows:
                if hasattr(om.flows[i, o], 'gas_grid_par') and om.flows[i,o].gas_grid_par == 1:
                    if any(substring in str(i) for substring in ['gas']):
                        flows_gas_1[(i, o)] = om.flows[i, o]
                    if any(substring in str(i) for substring in ['hydrogen']):
                        flows_h2_1[(i, o)] = om.flows[i, o]
                
                if hasattr(om.flows[i, o], 'gas_grid_par') and om.flows[i,o].gas_grid_par == 6:
                    if any(substring in str(i) for substring in ['gas']):
                        flows_gas_6[(i, o)] = om.flows[i, o]
                    if any(substring in str(i) for substring in ['hydrogen']):
                        flows_h2_6[(i, o)] = om.flows[i, o]
                        
                if hasattr(om.flows[i, o], 'gas_grid_par') and om.flows[i,o].gas_grid_par == 16:
                    if any(substring in str(i) for substring in ['gas']):
                        flows_gas_16[(i, o)] = om.flows[i, o]
                    if any(substring in str(i) for substring in ['hydrogen']):
                        flows_h2_16[(i, o)] = om.flows[i, o]
                        
                if hasattr(om.flows[i, o], 'gas_grid_par') and om.flows[i,o].gas_grid_par == 19:
                    if any(substring in str(i) for substring in ['gas']):
                        flows_gas_19[(i, o)] = om.flows[i, o]
                    if any(substring in str(i) for substring in ['hydrogen']):
                        flows_h2_19[(i, o)] = om.flows[i, o]               
                    
                    
        else:
            for (i, o) in om.flows:
                if not hasattr(flows[i, o], 'gas_grid_par'):
                    raise ValueError(f'Flow with source: {i.label} and target: {o.label} has no attribute w')

        dict_flow = {}
        for (i, o) in flows:
            if hasattr(flows[i, o], 'gas_grid_par'):
                if flows[i, o].link_par in dict_flow:
                    dict_flow[flows[i, o].gas_grid_par].append(flows[i, o])
                else:
                    dict_flow[flows[i, o].gas_grid_par] = [flows[i, o]]
                    
                    
        gas_factor = 2.68
        h2_factor= 8.92
        def link_1(om, p, t):            
            lhs_gas = sum(om.flow[i, o, p, t] for (i, o) in flows_gas_1) * gas_factor
            lhs_h2 = sum(om.flow[i, o, p, t] for (i, o) in flows_gas_1) * h2_factor
            rhs= 59       
            return lhs_gas + lhs_h2 <= rhs

        om.link_gas_h2_1 = po.Constraint(om.TIMEINDEX, rule=link_1)
        
        def link_6(om, p, t):            
            lhs_gas = sum(om.flow[i, o, p, t] for (i, o) in flows_gas_6) * gas_factor
            lhs_h2 = sum(om.flow[i, o, p, t] for (i, o) in flows_gas_6) * h2_factor
            rhs= 109       
            return lhs_gas + lhs_h2 <= rhs

        om.link_gas_h2_6 = po.Constraint(om.TIMEINDEX, rule=link_6)
        
        def link_16(om, p, t):            
            lhs_gas = sum(om.flow[i, o, p, t] for (i, o) in flows_gas_16) * gas_factor
            lhs_h2 = sum(om.flow[i, o, p, t] for (i, o) in flows_gas_16) * h2_factor
            rhs= 44       
            return lhs_gas + lhs_h2 <= rhs

        om.link_gas_h2_16 = po.Constraint(om.TIMEINDEX, rule=link_16)
        
        def link_19(om, p, t):            
            lhs_gas = sum(om.flow[i, o, p, t] for (i, o) in flows_gas_19) * gas_factor
            lhs_h2 = sum(om.flow[i, o, p, t] for (i, o) in flows_gas_19) * h2_factor
            rhs= 140.8       
            return lhs_gas + lhs_h2 <= rhs
        
        om.link_gas_h2_19 = po.Constraint(om.TIMEINDEX, rule=link_19)
        
        return om

    om  = gas_h2_supply(om)

    #%% Vincolo supply

    def supply_con (om):
        
        flows_supply ={}
        
        for (i, o) in om.flows:
                    if hasattr(om.flows[i, o], 'supply_constraint'):
                        flows_supply[(i, o)] = om.flows[i, o]
              
        def rule_supply_con(om, pp):
            lhs1 = 0.0
            TIMEINDEX_MOD = [(p, t) for (p, t) in om.TIMEINDEX if p == pp]
            for p, t in TIMEINDEX_MOD:
                lhs = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_supply)
                lhs1 += lhs
            
            expr= lhs1 - 40000
            return expr <= 0
        
        om.supply_const = po.Constraint(om.PERIODS, rule=rule_supply_con)
        return om
    
    om = supply_con(om)    

    #%% Vincolo Rampe

    # def con_PV_ramp (om,limit_ramp_pv, limit_ramp_wind, limit_ramp_wind_off):
    #                 flows_pv_ramp ={}
    #                 flows_wind_ramp ={}
    #                 flows_wind_ramp_off ={}
                   
    #                 #p= len(om.es.periods)-1
    #                 for (i, o) in om.flows:
    #                             if hasattr(om.flows[i, o], 'inv_ramp') and om.flows[i,o].inv_ramp == 1:
    #                                 flows_pv_ramp[(i, o)] = om.flows[i, o]
    #                 for (i, o) in om.flows:
    #                             if hasattr(om.flows[i, o], 'inv_ramp') and om.flows[i,o].inv_ramp == 2:
    #                                 flows_wind_ramp[(i, o)] = om.flows[i, o]
    #                 for (i, o) in om.flows:
    #                             if hasattr(om.flows[i, o], 'inv_ramp') and om.flows[i,o].inv_ramp == 4:
    #                                 flows_wind_ramp_off[(i, o)] = om.flows[i, o]
                       
    #                 def rule_ramp_pv(om, p):
                    
    #                     lhs = sum(om.InvestmentFlowBlock.invest[i, o, p] for (i, o) in flows_pv_ramp)
                           
    #                         #om.repetition_per_hour[t]
    #                     rhs = limit_ramp_pv[p]
                           
                       
    #                     expr= lhs - rhs  
                       
    #                     return expr <= 0
                       
                   
    #                 om.const_PV_ramp = po.Constraint(om.PERIODS, rule=rule_ramp_pv)
                    
                    
    #                 def rule_ramp_wind(om, p):
                    
    #                     lhs = sum(om.InvestmentFlowBlock.invest[i, o, p] for (i, o) in flows_wind_ramp)
                           
    #                         #om.repetition_per_hour[t]
    #                     rhs = limit_ramp_wind[p]
                           
                       
    #                     expr= lhs - rhs  
                       
    #                     return expr <= 0
                       
                   
    #                 om.const_Wind_ramp = po.Constraint(om.PERIODS, rule=rule_ramp_wind)
                    
                    
    #                 def rule_ramp_wind_off(om, p):
                    
    #                     lhs = sum(om.InvestmentFlowBlock.invest[i, o, p] for (i, o) in flows_wind_ramp_off)
                           
    #                         #om.repetition_per_hour[t]
    #                     rhs = limit_ramp_wind_off[p]
                           
                       
    #                     expr= lhs - rhs  
                       
    #                     return expr <= 0
                       
                   
    #                 om.const_Wind_ramp_off = po.Constraint(om.PERIODS, rule=rule_ramp_wind_off)
                    
                    
                   
    #                 return om

    # limit_ramp_pv =[40,40,60,60,60,60,60]
    # limit_ramp_wind =[15,15,30,30,30,30,30]
    # limit_ramp_wind_off = [5,5,5,10,10,10,10]
    # om = con_PV_ramp(om,limit_ramp_pv,limit_ramp_wind,limit_ramp_wind_off)



    def con_PV_WD_ELC_max_level (om):
                    
                    periodi_CO2 = sorted(list(set(om.es.periods.keys())))
                    om.PERIODS_MIN_INV = po.Set(
                            initialize= periodi_CO2[3:]
                        )
                    
                    flows_pv_ramp ={}
                    flows_wind_ramp ={}
                    flows_elec_ramp ={}
                    flows_wind_off_ramp={}
                   
                    for (i, o) in om.flows:
                                if hasattr(om.flows[i, o], 'inv_ramp') and om.flows[i,o].inv_ramp == 1:
                                    flows_pv_ramp[(i, o)] = om.flows[i, o]
                    for (i, o) in om.flows:
                                if hasattr(om.flows[i, o], 'inv_ramp') and om.flows[i,o].inv_ramp == 2:
                                    flows_wind_ramp[(i, o)] = om.flows[i, o]
                    for (i, o) in om.flows:
                                if hasattr(om.flows[i, o], 'inv_ramp') and om.flows[i,o].inv_ramp == 3:
                                    flows_elec_ramp[(i, o)] = om.flows[i, o]
                    for (i, o) in om.flows:
                                if hasattr(om.flows[i, o], 'inv_ramp') and om.flows[i,o].inv_ramp == 4:
                                    flows_wind_off_ramp[(i, o)] = om.flows[i, o]
                       
                    def rule_min_inv_pv(om, p):
                    
                        lhs = sum(om.InvestmentFlowBlock.total[i, o, p] for (i, o) in flows_pv_ramp)
                        rhs = sum(om.InvestmentFlowBlock.total[i, o, p-1] for (i, o) in flows_pv_ramp)
                        expr= rhs - lhs
                       
                        return expr <= 0
                   
                    om.const_PV_min_inv = po.Constraint(om.PERIODS_MIN_INV, rule=rule_min_inv_pv)
                   
                    
                    def rule_min_inv_wind(om, p):
                    
                        lhs = sum(om.InvestmentFlowBlock.total[i, o, p] for (i, o) in flows_wind_ramp)
                        rhs = sum(om.InvestmentFlowBlock.total[i, o, p-1] for (i, o) in flows_wind_ramp)
                       
                   
                        expr= rhs - lhs
                   
                        return expr <= 0                      
                   
                    om.const_Wind_min_inv = po.Constraint(om.PERIODS_MIN_INV, rule=rule_min_inv_wind)
                    
                    def rule_min_inv_wind_off(om, p):
                    
                        lhs = sum(om.InvestmentFlowBlock.total[i, o, p] for (i, o) in flows_wind_off_ramp)                         
                        rhs = sum(om.InvestmentFlowBlock.total[i, o, p-1] for (i, o) in flows_wind_off_ramp)
                       
                   
                        expr= rhs - lhs
                   
                        return expr <= 0
                       
                   
                    om.const_Wind_off_min_inv = po.Constraint(om.PERIODS_MIN_INV, rule=rule_min_inv_wind_off)
                    
                    
                    def rule_min_inv_elec(om, p):
                    
                        lhs = sum(om.InvestmentFlowBlock.total[i, o, p] for (i, o) in flows_elec_ramp)
                        rhs = sum(om.InvestmentFlowBlock.total[i, o, p-1] for (i, o) in flows_elec_ramp)
                           
                       
                        expr= rhs - lhs
                       
                        return expr <= 0
                       
                   
                    om.const_elec_min_inv = po.Constraint(om.PERIODS_MIN_INV, rule=rule_min_inv_elec)
                   
                    return om


    om = con_PV_WD_ELC_max_level(om)
    
    #%% Import constraint
    def con_limit_import (om):
                    flows_import_EE ={}
                    flows_import_gas ={}
                    flows_import_LF ={}
                    flows_import_h2 ={}
                    flows_import_biomass ={}
                    
                    for (i, o) in om.flows:
                                if hasattr(om.flows[i, o], 'supply_2050_2055') and om.flows[i,o].supply_2050_2055 == 1:
                                    flows_import_gas[(i, o)] = om.flows[i, o]
                    for (i, o) in om.flows:
                                if hasattr(om.flows[i, o], 'supply_2050_2055') and om.flows[i,o].supply_2050_2055 == 2:
                                    flows_import_EE[(i, o)] = om.flows[i, o]
                    for (i, o) in om.flows:
                                if hasattr(om.flows[i, o], 'supply_2050_2055') and om.flows[i,o].supply_2050_2055 == 3:
                                    flows_import_LF[(i, o)] = om.flows[i, o]
                    for (i, o) in om.flows:
                                if hasattr(om.flows[i, o], 'supply_2050_2055') and om.flows[i,o].supply_2050_2055 == 4:
                                    flows_import_biomass[(i, o)] = om.flows[i, o]
                    for (i, o) in om.flows:
                                if hasattr(om.flows[i, o], 'supply_2050_2055') and om.flows[i,o].supply_2050_2055 == 5:
                                    flows_import_h2[(i, o)] = om.flows[i, o]
                       
                    def rule_import_2050_EE(om):
                    
                        lhs_2050 = 0.0
                        lhs_2055 = 0.0
                        TIMEINDEX_MOD_2 = [(p, t) for (p, t) in om.TIMEINDEX if p == om.PERIODS.at(-1)]
                        TIMEINDEX_MOD_1 = [(p, t) for (p, t) in om.TIMEINDEX if p == om.PERIODS.at(-2)]
                        for p, t in TIMEINDEX_MOD_1:
                            lhs_2055_1 = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_import_EE)
                            lhs_2055 += lhs_2055_1
                        for p, t in TIMEINDEX_MOD_2:
                            lhs_2050_1 = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_import_EE)
                            lhs_2050 += lhs_2050_1
                             
                        expr = lhs_2055-lhs_2050
                        return expr <= 0
                       
                   
                    om.const_import_2050_EE = po.Constraint( rule=rule_import_2050_EE)
                    
                    def rule_import_2050_gas(om):
                    
                        lhs_2050 = 0.0
                        lhs_2055 = 0.0
                        TIMEINDEX_MOD_2 = [(p, t) for (p, t) in om.TIMEINDEX if p == om.PERIODS.at(-1)]
                        TIMEINDEX_MOD_1 = [(p, t) for (p, t) in om.TIMEINDEX if p == om.PERIODS.at(-2)]
                        for p, t in TIMEINDEX_MOD_1:
                            lhs_2055_1 = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_import_gas)
                            lhs_2055 += lhs_2055_1
                        for p, t in TIMEINDEX_MOD_2:
                            lhs_2050_1 = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_import_gas)
                            lhs_2050 += lhs_2050_1
                             
                        expr = lhs_2055-lhs_2050
                        return expr <= 0                       
                   
                    om.const_import_2050_gas = po.Constraint( rule=rule_import_2050_gas)
                    
                    def rule_import_2050_LF(om):
                    
                        lhs_2050 = 0.0
                        lhs_2055 = 0.0
                        TIMEINDEX_MOD_2 = [(p, t) for (p, t) in om.TIMEINDEX if p == om.PERIODS.at(-1)]
                        TIMEINDEX_MOD_1 = [(p, t) for (p, t) in om.TIMEINDEX if p == om.PERIODS.at(-2)]
                        for p, t in TIMEINDEX_MOD_1:
                            lhs_2055_1 = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_import_LF)
                            lhs_2055 += lhs_2055_1
                        for p, t in TIMEINDEX_MOD_2:
                            lhs_2050_1 = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_import_LF) 
                            lhs_2050 += lhs_2050_1
                             
                        expr = lhs_2055-lhs_2050
                        return expr <= 0                       
                   
                    om.const_import_2050_LF = po.Constraint(rule=rule_import_2050_LF)                    
                    
                    def rule_import_2050_biomass(om):
                    
                        lhs_2050 = 0.0
                        lhs_2055 = 0.0
                        TIMEINDEX_MOD_2 = [(p, t) for (p, t) in om.TIMEINDEX if p == om.PERIODS.at(-1)]
                        TIMEINDEX_MOD_1 = [(p, t) for (p, t) in om.TIMEINDEX if p == om.PERIODS.at(-2)]
                        for p, t in TIMEINDEX_MOD_1:
                            lhs_2055_1 = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_import_biomass)
                            lhs_2055 += lhs_2055_1
                        for p, t in TIMEINDEX_MOD_2:
                            lhs_2050_1 = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_import_biomass)
                            lhs_2050 += lhs_2050_1
                             
                        expr = lhs_2055-lhs_2050
                        return expr <= 0                       
                   
                    om.const_import_2050_biomass = po.Constraint(rule=rule_import_2050_biomass)                    
                    
                    def rule_import_2050_h2(om):
                    
                        lhs_2050 = 0.0
                        lhs_2055 = 0.0
                        TIMEINDEX_MOD_2 = [(p, t) for (p, t) in om.TIMEINDEX if p == om.PERIODS.at(-1)]
                        TIMEINDEX_MOD_1 = [(p, t) for (p, t) in om.TIMEINDEX if p == om.PERIODS.at(-2)]
                        for p, t in TIMEINDEX_MOD_1:
                            lhs_2055_1 = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_import_h2)
                            lhs_2055 += lhs_2055_1
                        for p, t in TIMEINDEX_MOD_2:
                            lhs_2050_1 = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_import_h2)
                            lhs_2050 += lhs_2050_1
                             
                        expr = lhs_2055-lhs_2050
                        return expr <= 0
                       
                   
                    om.const_import_2050_h2 = po.Constraint(rule=rule_import_2050_h2)                   

                    return om

    om = con_limit_import(om)

    #%%  CO2 Sink constrain
    def CO2_sink_con (om):
        
        flows_CO2_sink ={}
        
        for (i, o) in om.flows:
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 15:
                        flows_CO2_sink[(i, o)] = om.flows[i, o]
          
        def rule_CO2_sink_con(om, pp):
            lhs1 = 0.0
            TIMEINDEX_MOD = [(p, t) for (p, t) in om.TIMEINDEX if p == pp]
            for p, t in TIMEINDEX_MOD:
                lhs = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_CO2_sink)
                lhs1 += lhs                
            
            expr= lhs1 - 16000  #20000
            
            return expr <= 0
        
        om.CO2_sink_const = po.Constraint(om.PERIODS, rule=rule_CO2_sink_con)
        
        return om

    om = CO2_sink_con(om)
    
    def CO2_sink_con_2030 (om):
        
        flows_CO2_sink ={}
        
        #p= len(om.es.periods)-1
        for (i, o) in om.flows:
                    if hasattr(om.flows[i, o], 'emission_constraint') and om.flows[i,o].emission_constraint == 15:
                        flows_CO2_sink[(i, o)] = om.flows[i, o]
        
              
        def rule_CO2_sink_con_2030(om):
            lhs1 = 0.0
            pp=1
            TIMEINDEX_MOD = [(p, t) for (p, t) in om.TIMEINDEX if p == pp]
            for p, t in TIMEINDEX_MOD:
                lhs = sum(om.flow[i, o, p, t] *om.repetition_per_hour[t] for (i, o) in flows_CO2_sink)
                lhs1 += lhs                
            
            expr= lhs1 - 4000  #20000
            
            return expr <= 0            
        
        om.CO2_sink_const_2030 = po.Constraint(rule=rule_CO2_sink_con_2030)
        
        return om

    om = CO2_sink_con_2030(om)

    #%%
             
    print("********************************************************")
    logging.info('OPTIMIZATION')

    opt = po.SolverFactory('gurobi')
    #om.params.Method = 0
    opt.options["Method"] = 2
    opt.options["Crossover"] = 0
    opt.options["Presolve"] = 2
    #opt.options["NumericFocus"]=1
    #opt.options["ScaleFlag"] = 2


    from datetime import datetime
    # Get the current date and time
    now = datetime.now()
    # Create a datestring in the specified format
    datestring = now.strftime('%Y%m%d_%H%M')
    # Define the options
    T1 = InputData
    sim_name = f'Results/Blend_{blend_folder}_h2Inv_{hydrogen_invest_case}_EEInv_{ee_grid_invest_case}_invco2pip{co2_grid_invest_case}_co2cum_{comulative_case}_discount_{discount_rate}_mulex_{multiplier_excess}/Blend_{blend_folder}_h2Inv_{hydrogen_invest_case}_EEInv_{ee_grid_invest_case}_invco2pip{co2_grid_invest_case}_co2cum_{comulative_case}.txt'
    opt.options["LogFile"] = sim_name
    opt.solve(om, tee= True)

                            # solving the linear problem using the given solver ('cbc')
    #om.solve(solver='gurobi', tee= True)
    print("********************************************************")
    logging.info('Processing results')
    results, result_storage= solph.processing.results(om,remove_last_time_point=True)
    result = solph.processing.convert_keys_to_strings(results)

    results_storage = solph.processing.convert_keys_to_strings(result_storage)

    #%% Export data

    # Save the data to a file
    filename =  f'Results/Blend_{blend_folder}_h2Inv_{hydrogen_invest_case}_EEInv_{ee_grid_invest_case}_invco2pip{co2_grid_invest_case}_co2cum_{comulative_case}_discount_{discount_rate}_mulex_{multiplier_excess}/results.pkl'   #'results.pkl'  
    with open(filename, 'wb') as f:
        pickle.dump(result, f)

    filename = f'Results/Blend_{blend_folder}_h2Inv_{hydrogen_invest_case}_EEInv_{ee_grid_invest_case}_invco2pip{co2_grid_invest_case}_co2cum_{comulative_case}_discount_{discount_rate}_mulex_{multiplier_excess}/result_storage.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results_storage, f)
