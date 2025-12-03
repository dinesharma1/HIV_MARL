import numpy as np
import pandas as pd
import os

# INFO: Variables used in this file
num_risk = 3

age_groups = 88
number_of_risk_groups = 3
number_of_compartments = 22

#Change this to read from excel? or do this only once not in every file

def choose_jur():

    #CA, FL, Georgia, New York, OHio, Texas, Loiusiana, New Jersey, Penn, Arixona
    #all_jurisdictions = [6001, 6037, 6059, 6065, 6067, 6071, 6073, 6]
    """
    all_jurisdictions = [6001, 6037, 6059, 6065, 6067, 6071, 6073, 6,
                        12011, 12031, 12057, 12086, 12095, 12099, 12103, 12,
                        13067, 13089, 13121, 13135, 13,
                        36005, 36047, 36061, 36081, 36,
                        39035, 39049, 39061, 39,
                        48029, 48113, 48201, 48439, 48453, 48,
                        22033, 22071, 22,
                        34013, 34017, 34,
                        42101, 42,
                        4013, 4]
    #"""

    
    #96 jurisdictions considered #alphabetical order
    #"""
    all_jurisdictions = [
                        1, 2,                                                   #Alabama. Alaska
                        #3 - NOT PRESENT IN SURVEY DATA
                        4, 4013,                                                #Arizona
                        5,                                                      #Arkansas
                        6, 6001, 6037, 6059, 6065, 6067, 6071, 6073,            #California
                        #7 - NOT PRESENT IN SURVEY DATA
                        8, 9, 10, 11,                                           #Colorado, Connecticut, Delaware, DC
                        12, 12011, 12031, 12057, 12086, 12095, 12099, 12103,    #Florida
                        13, 13067, 13089, 13121, 13135,                         #Georgia
                        #14 - NOT PRESENT IN SURVEY DATA
                        15, 16,                                                 #Hawaii, Idaho
                        17, 17031,                                              #Illinois
                        18, 18097,                                              #Indiana
                        19, 20, 21,                                             #Iowa, Kansas, Kentucky
                        22, 22033, 22071,                                       #Louisiana
                        23,                                                     #Maine
                        24, 24031, 24033, #24510,                               #Maryland
                        25,                                                     #Massachusetts
                        26, 26163,                                              #Michigan
                        27, 28, 29, 30, 31,                                     #Minnesota, Mississippi, Missouri, Montana, Nebraska
                        32, 32003,                                              #Nevada
                        #33 - NOT INCLUDED IN 96 JURI #New Hampshire
                        34, 34013, 34017,                                       #New Jersey
                        35,                                                     #New Mexico
                        36, 36005, 36047, 36061, 36081,                         #New York
                        37, 37119,                                              #North Carolina
                        38,                                                     #North Dakota
                        39, 39035, 39049, 39061,                                #Ohio
                        40, 41,                                                 #Oklahoma, Oregon
                        42, 42101,                                              #Pennsylvania
                        #43 - NOT PRESENT IN SURVEY DATA
                        44, 45, 46,                                             #Rhode Island, SOth Carlolina, South Dakota
                        47, 47157,                                              #Tennessee
                        48, 48029, 48113, 48201, 48439, 48453,                  #Texas
                        49, 50, 51,                                             #Utah, Vermont, Virginia
                        #52 - NOT PRESENT IN SURVEY DATA
                        53, 53033,                                              #Washington
                        54, 55, 56                                              #West Virginia, Wisconsin, Wyoming
                        ] 
    #"""

    #24510 #Maryland (Baltimore City)- Causing overflow errors
    #all_jurisdictions = [24, 24031, 24033,] #24510
    #all_jurisdictions = [24510]
    #all_jurisdictions = [6, 6001, 6037, 6059, 6065, 6067, 6071, 6073]
    return(all_jurisdictions)

all_jurisdictions = choose_jur()

num_jur = len(all_jurisdictions)

# INFO: This code block initializes an multi-dimensional numpy array and fills it with the respective 
#       jurisdictional values from input excel data files. The shape of this array is (num_jur, 3, 88, 22)

data_array_cluster = np.zeros((num_jur, number_of_risk_groups, age_groups, number_of_compartments))


for jur in range(len(all_jurisdictions)):

    #cwd = os.getcwd() #c:\Users\Sonza\workspace3\HIV_MARL_96
    #print(cwd)
    df1 = pd.read_excel('./input_files/Jurisdiction_pop_dist.xlsx', sheet_name=str(all_jurisdictions[jur]), index_col=0)
    data_array_cluster[jur] = df1.iloc[:,3:].to_numpy().reshape(3,88,22)

# INFO: This line calculates the total population in each jurisdiction
total_pop = np.apply_over_axes(np.sum, data_array_cluster, [1,2,3]).reshape(num_jur,)
total_pop = np.array([int(x) for x in total_pop])

# INFO: This line calculates the total MSM population in each jurisdiction
total_msm = list(np.apply_over_axes(np.sum, data_array_cluster, [2,3]).reshape(num_jur,3)[:,2])
total_msm = np.array([int(x) for x in total_msm])

df_prep = pd.read_excel('./input_files/Prep values.xlsx')

df_prep_clus = df_prep.loc[df_prep['FIPS'].isin(all_jurisdictions)]

jur_name = df_prep_clus['JUR'].to_list()

df_prep_clus['Total MSM'] = total_msm

prep_values = np.zeros((num_jur,num_risk))

prep_rates_clus = df_prep_clus['Prep'].values

prep_eligible = df_prep_clus['PrEP Eligible'].values

prep_values[:,2] = prep_rates_clus/100

# INFO: This code block gets the budget values for each jurisdiction from input excel data file
df_budget = pd.read_excel('./input_files/Budget_per_juri.xlsx')
df_budget_exact = df_budget.loc[pd.Index(df_budget['FIPS']).get_indexer(all_jurisdictions)]
budget_values = df_budget_exact['Budget'].values
print(budget_values)

# INFO: This code block gets the proportion unware and proportion ART at time 0 for each jurisdiction from input excel data file

df_hm = pd.read_excel('./input_files/CareContinuum-by_jur_5_28_25_SS.xlsx',sheet_name='jur_specific_care_cont_hm')
df_hm_exact = df_hm.loc[pd.Index(df_hm['FIPS']).get_indexer(all_jurisdictions)]
p_unaware_0_hm = df_hm_exact['Unaware'].values
p_art_0_hm = df_hm_exact['ART'].values

#print(all_jurisdictions)
#print(p_unaware_0_hm)

df_hf = pd.read_excel('./input_files/CareContinuum-by_jur_5_28_25_SS.xlsx',sheet_name='jur_specific_care_cont_hf')
df_hf_exact = df_hf.loc[pd.Index(df_hf['FIPS']).get_indexer(all_jurisdictions)]
p_unaware_0_hf = df_hf_exact['Unaware'].values
p_art_0_hf = df_hf_exact['ART'].values


df_msm = pd.read_excel('./input_files/CareContinuum-by_jur_5_28_25_SS.xlsx',sheet_name='jur_specific_care_cont_msm')
df_msm_exact = df_msm.loc[pd.Index(df_msm['FIPS']).get_indexer(all_jurisdictions)]
p_unaware_0_msm = df_msm_exact['Unaware'].values
p_art_0_msm = df_msm_exact['ART'].values

p_unaware_0 = np.transpose(np.vstack((p_unaware_0_hm, p_unaware_0_hf, p_unaware_0_msm)))
p_art_0 = np.transpose(np.vstack((p_art_0_hm, p_art_0_hf, p_art_0_msm)))

#print(p_unaware_0)
#print(p_art_0)
#exit(0)


