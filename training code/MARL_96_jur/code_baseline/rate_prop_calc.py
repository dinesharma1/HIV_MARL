
import numpy as np
import pandas as pd
from choose_jurisdiction import *

import warnings
warnings.filterwarnings("error",category=RuntimeWarning)

#warnings.resetwarnings() #reset behavior of warnings if needed

import import_ipynb
from read_files import read_new_inf_input

#from ipynb.fs.full.reading_files import read_new_inf_input

age_begin = 13
age_end = 100
risk_groups = ['HM', 'HF', 'MSM']
num_risk = 3

#all_jur = 10
#age_groups = 88
number_of_risk_groups = 3
number_of_compartments = 22
dt = 1/12

num_age = 88
num_comp = number_of_compartments-2

#prep_efficiency = 0.99

unaware_index = (1,5,9,13,17)
aware_no_care_index = (2,6,10,14,18)
ART_VLS_index = (3,4,7,8,11,12,15,16,19,20)
VLS_index = (4,8,12,16,20)


#pop_growth_rate = 0

gamma = np.array([[0.5,0.5,0.5,0.5,1],
                  [0.5,0.5,0.5,0.5,1],
                  [0.5,0.5,0.5,0.5,1]])

scaling_factor_dropout = np.array([[1,1,1,1,1,1,1,1,0,0],
                                   [1,1,1,1,1,1,1,1,0,0],
                                   [1,1,1,1,1,1,1,1,0,0]])

all_jurisdictions = choose_jur()

num_jur = len(all_jurisdictions)

new_infections_data = read_new_inf_input(age_begin, age_end, risk_groups)

mixing_excel = './input_files/JURI_mixing_weightedBydistance-6-3-2021.xlsx'

mixing_df_hm = pd.read_excel(mixing_excel, sheet_name='HETM_mixing')
mixing_df_hf = pd.read_excel(mixing_excel, sheet_name='HETF_mixing')
mixing_df_msm = pd.read_excel(mixing_excel, sheet_name='MSM_mixing')

#Sonza changes

mixing_array = ['FIPS'] + all_jurisdictions
#print(mixing_array)
#exit(0)

mixing_df_hm = mixing_df_hm[mixing_array]
mixing_df_hf = mixing_df_hf[mixing_array]
mixing_df_msm = mixing_df_msm[mixing_array]
#"""

#CA
"""
mixing_df_hm = mixing_df_hm[['FIPS', 6001,6037,6059,6065,6067,6071,6073,6]]
mixing_df_hf = mixing_df_hf[['FIPS', 6001,6037,6059,6065,6067,6071,6073,6]]
mixing_df_msm = mixing_df_msm[['FIPS', 6001,6037,6059,6065,6067,6071,6073,6]]
#"""

#FL
"""
mixing_df_hm = mixing_df_hm[['FIPS', 12011, 12031, 12057, 12086, 12095, 12099, 12103, 12]]
mixing_df_hf = mixing_df_hf[['FIPS', 12011, 12031, 12057, 12086, 12095, 12099, 12103, 12]]
mixing_df_msm = mixing_df_msm[['FIPS', 12011, 12031, 12057, 12086, 12095, 12099, 12103, 12]]
"""

#CA and FL
"""
mixing_df_hm = mixing_df_hm[['FIPS', 6001,6037,6059,6065,6067,6071,6073,6, 12011, 12031, 12057, 12086, 12095, 12099, 12103, 12]]
mixing_df_hf = mixing_df_hf[['FIPS', 6001,6037,6059,6065,6067,6071,6073,6, 12011, 12031, 12057, 12086, 12095, 12099, 12103, 12]]
mixing_df_msm = mixing_df_msm[['FIPS', 6001,6037,6059,6065,6067,6071,6073,6, 12011, 12031, 12057, 12086, 12095, 12099, 12103, 12]]
#"""

#CA and FL (specific juris)
"""
mixing_df_hm = mixing_df_hm[['FIPS', 6001,6037,6059,6065,6067,6071,6073, 12011, 12031, 12057, 12086, 12095, 12099, 12103]]
mixing_df_hf = mixing_df_hf[['FIPS', 6001,6037,6059,6065,6067,6071,6073, 12011, 12031, 12057, 12086, 12095, 12099, 12103]]
mixing_df_msm = mixing_df_msm[['FIPS', 6001,6037,6059,6065,6067,6071,6073, 12011, 12031, 12057, 12086, 12095, 12099, 12103]]
#"""

#CA and FL (rest of state)
"""
mixing_df_hm = mixing_df_hm[['FIPS', 6, 12, 13, 36]]
mixing_df_hf = mixing_df_hf[['FIPS', 6, 12, 13, 36]]
mixing_df_msm = mixing_df_msm[['FIPS', 6, 12, 13, 36]]
#"""

"""
mixing_df_hm = mixing_df_hm[['FIPS', 6001, 6037, 6059, 6065, 6067, 6071, 6073, 6,
                    12011, 12031, 12057, 12086, 12095, 12099, 12103, 12,
                    13067, 13089, 13121, 13135, 13,
                    36005, 36047, 36061, 36081, 36,
                    39035, 39049, 39061, 39,
                    48029, 48113, 48201, 48439, 48453, 48,
                    22033, 22071, 22,
                    24031, 24033, 24510, 24,
                    34013, 34017, 34,
                    42101, 42,
                    4013, 4]]
mixing_df_hf = mixing_df_hf[['FIPS', 6001, 6037, 6059, 6065, 6067, 6071, 6073, 6,
                    12011, 12031, 12057, 12086, 12095, 12099, 12103, 12,
                    13067, 13089, 13121, 13135, 13,
                    36005, 36047, 36061, 36081, 36,
                    39035, 39049, 39061, 39,
                    48029, 48113, 48201, 48439, 48453, 48,
                    22033, 22071, 22,
                    24031, 24033, 24510, 24,
                    34013, 34017, 34,
                    42101, 42,
                    4013, 4]]
mixing_df_msm = mixing_df_msm[['FIPS', 6001, 6037, 6059, 6065, 6067, 6071, 6073, 6,
                    12011, 12031, 12057, 12086, 12095, 12099, 12103, 12,
                    13067, 13089, 13121, 13135, 13,
                    36005, 36047, 36061, 36081, 36,
                    39035, 39049, 39061, 39,
                    48029, 48113, 48201, 48439, 48453, 48,
                    22033, 22071, 22,
                    24031, 24033, 24510, 24,
                    34013, 34017, 34,
                    42101, 42,
                    4013, 4]]
#"""

mixing_hm = mixing_df_hm.loc[mixing_df_hm['FIPS'].isin(all_jurisdictions)]
mixing_hf = mixing_df_hf.loc[mixing_df_hf['FIPS'].isin(all_jurisdictions)]
mixing_msm = mixing_df_msm.loc[mixing_df_msm['FIPS'].isin(all_jurisdictions)]

hm_array = mixing_hm.values[:,1:]
hf_array = mixing_hf.values[:,1:]
msm_array = mixing_msm.values[:,1:]

hm_sum = np.sum(hm_array, axis=1)
hf_sum = np.sum(hf_array, axis=1)
msm_sum = np.sum(msm_array, axis=1)

#normalization
hm_array_scaled = hm_array / hm_sum[:,np.newaxis]
hf_array_scaled = hf_array / hf_sum[:,np.newaxis]
msm_array_scaled = msm_array / msm_sum[:,np.newaxis]

mixing_matrix = np.zeros((num_risk,len(all_jurisdictions), len(all_jurisdictions)))

mixing_matrix[0,:,:] = hm_array_scaled[:,:]
mixing_matrix[1,:,:] = hf_array_scaled[:,:]
mixing_matrix[2,:,:] = msm_array_scaled[:,:]

#np.savetxt("mixing_hm.csv", mixing_matrix[0,:,:], delimiter=",")
#np.savetxt("mixing_hf.csv", mixing_matrix[1,:,:], delimiter=",")
#np.savetxt("mixing_msm.csv", mixing_matrix[2,:,:], delimiter=",")

#exit(0)

def new_infections_per_month(num_jur, data_array, new_infections_data, M_x1_y1_i, prep_risk):
    
    risk_mat = new_infections_data["sex_mixing"].copy()[0:num_risk,0:num_risk]
    age_mat = new_infections_data["age_mixing_final_mat"].copy()
    
    I = data_array[:,:,:,1:21]
    N = np.sum(data_array[:,:,:,0:21], axis=3)
    d_x1_y1 = new_infections_data["num_partner_risk_casual"].copy()+new_infections_data["num_partner_risk_casual_only"].copy()
    sus_x1_y1 = data_array[:,:,:,0]
    mat_vector = np.repeat(age_mat[:,np.newaxis,:,:],num_risk, axis = 1) * risk_mat[:,:,np.newaxis, np.newaxis]
    I_N_vector = I / N[:,:,:,np.newaxis]
    I_N_mult_vector = mat_vector[np.newaxis,:,:,:,:,np.newaxis]*I_N_vector[:,np.newaxis,:,np.newaxis,:,:]
    Q_inner_vector = np.apply_over_axes(np.sum, I_N_mult_vector, [2,4]).reshape((num_jur, num_risk,num_age,num_comp))
    q_x_y_i_vector = d_x1_y1[np.newaxis,:,np.newaxis,np.newaxis]*dt*Q_inner_vector
    q_mix_vector = np.zeros((num_jur,num_jur,num_risk,num_age,num_comp))
    for risk in range(num_risk):
        q_mix_vector[:,:,risk,:,:] = mixing_matrix[risk][:,:,np.newaxis,np.newaxis]*q_x_y_i_vector[:,risk,:,:][np.newaxis,:,:,:]
    q_mix_sum_vector = np.sum(q_mix_vector, axis = 1)
    M_power_vector = M_x1_y1_i[np.newaxis,:,:,:]**q_mix_sum_vector
    
    """
    print("reached try catch 1")
    try:
        #print(q_mix_sum_vector.shape) #(50, 3, 88, 20)
        #print(q_mix_sum_vector.flatten())
        #print(M_x1_y1_i[np.newaxis,:,:,:].shape) #(1, 3, 88, 20)
        q_mix_sum_vector = np.round(q_mix_sum_vector,2)
        M_x1_y1_i = np.round(M_x1_y1_i,2)
        M_power_vector = np.power(M_x1_y1_i[np.newaxis,:,:,:],q_mix_sum_vector)
        print("Worked")
        #print(M_power_vector.shape) #(50, 3, 88, 20)
        #np.savetxt("m_power_vector.csv", q_mix_sum_vector.flatten(), delimiter=",")
        #np.savetxt("m_power_vector.csv", M_x1_y1_i[np.newaxis,:,:,:].flatten(), delimiter=",")
    except RuntimeWarning as error:
        print("Something went wrong")
        print('An exception occurred: {}'.format(error))
        exit(0)
        #breakpoint()
        #print(M_power_vector.shape)
        #np.savetxt("m_power_vector.csv", M_power_vector, delimiter=",")
        #np.savetxt("m_power_vector.csv", q_mix_sum_vector.flatten(), delimiter=",")
        #exit(0)
    #"""
    
    M_prod_vector = 1-np.prod(M_power_vector, axis = 3) 
    
    new_inf_per_month = sus_x1_y1*(1 - prep_risk[:,:,np.newaxis])*M_prod_vector
    
    return new_inf_per_month

def calculate_proportions(data_array, num_jur, number_of_risk_groups, unaware_index, aware_no_care_index, ART_VLS_index,VLS_index):
    
    plwh_risk = np.zeros((num_jur, number_of_risk_groups))
    unaware_risk = np.zeros((num_jur, number_of_risk_groups))
    aware_no_art_risk = np.zeros((num_jur, number_of_risk_groups))
    aware_art_vls_risk = np.zeros((num_jur, number_of_risk_groups))
    vls_risk = np.zeros((num_jur, number_of_risk_groups))
    
    for risk in range(number_of_risk_groups):
        plwh_risk[:,risk] = np.apply_over_axes(np.sum, data_array[:,risk,:,1:21], [1,2]).reshape(num_jur,)
#         print('risk')
#         print(plwh_risk[risk])
        unaware_risk[:,risk] = np.apply_over_axes(np.sum, data_array[:,risk,:,unaware_index], [0,2]).reshape(num_jur,)
#         print('unaware')
#         print(unaware_risk[risk])
        aware_no_art_risk[:,risk] = np.apply_over_axes(np.sum, data_array[:,risk,:,aware_no_care_index], [0,2]).reshape(num_jur,)
#         print('aware_no_art')
#         print(aware_no_art_risk[risk])
        aware_art_vls_risk[:,risk] = np.apply_over_axes(np.sum, data_array[:,risk,:,ART_VLS_index], [0,2]).reshape(num_jur,)
#         print('art vls')
#         print(aware_art_vls_risk[risk])

        vls_risk[:,risk] = np.apply_over_axes(np.sum, data_array[:,risk,:,VLS_index], [0,2]).reshape(num_jur,)
        
    
    total_pop = np.apply_over_axes(np.sum, data_array[:,:,:,0:21], [1,2,3]).reshape(num_jur,1)
    
    prevalence_prop = plwh_risk/total_pop
    
    unaware_prop = unaware_risk/plwh_risk
    aware_no_art_prop = aware_no_art_risk/plwh_risk
    aware_art_vls_prop = aware_art_vls_risk/plwh_risk
    vls_prop = vls_risk/plwh_risk
    
    return total_pop, np.round(prevalence_prop, 6), np.round(unaware_prop, 6), \
                np.round(aware_no_art_prop, 6), np.round(aware_art_vls_prop, 6), vls_prop

def diagnosis_rate(data_array, num_jur, a_unaware, unaware_index, number_of_risk_groups, new_inf_per_month, unaware_prop, death_per_month_risk_age_compartments):
    
    diagnosis_rate_risk = np.zeros((num_jur, number_of_risk_groups))
    
    a_unaware_t = np.round(a_unaware*dt, 6)

    for risk in range(len(risk_groups)):
        # new infectiion per month
        #print(np.shape(new_inf_per_month))
        A = np.sum(new_inf_per_month, axis=2)[:,risk]
        #print(A)

        # number of unaware population
        B = np.apply_over_axes(np.sum, data_array[:,risk,:,unaware_index], [0,2]).reshape(num_jur,)

        #number of unaware next time period
        # (current inf + new inf - total death)
        C = (np.apply_over_axes(np.sum, data_array[:,risk,:,1:21], [1,2]).reshape(num_jur,) + A - np.apply_over_axes(np.sum, death_per_month_risk_age_compartments[:,risk,:,1:21],[1,2]).reshape(num_jur,)) * (unaware_prop[:,risk] + a_unaware_t[:,risk])
        
        # total deaths in each compartment
        D = np.apply_over_axes(np.sum, death_per_month_risk_age_compartments[:,risk,:,unaware_index],[0,2]).reshape(num_jur,)
        #print(D)
        #raise Exception("Stopping for debugging")

        # number of people in unaware compartment
        E = np.sum(np.sum(data_array[:,risk,:, unaware_index],axis=2)*new_infections_data["testing_mult_fac_risk"][risk].reshape(5,1), axis=0)
        
        diagnosis_rate_risk[:,risk] = (A+B-C-D)/E
        
        diagnosis_rate_risk[:,risk][diagnosis_rate_risk[:,risk] < 0] = 0

    return diagnosis_rate_risk

def dropout_rate(num_jur, a_art, ART_VLS_index, diagnosis_rate_risk, ltc_risk, gamma, number_of_risk_groups, data_array, new_inf_per_month, unaware_prop, aware_no_art_prop, aware_art_vls_prop, death_per_month_risk_age_compartments):
    
    dropout_rate_risk = np.zeros((num_jur, number_of_risk_groups))
    
    a_art_t = np.round(a_art *dt, 6)
    gamma_t = np.round(gamma *dt, 6)
    
    for risk in range(len(risk_groups)):
       # total art vls pop 
        F = np.apply_over_axes(np.sum, data_array[:,risk,:,ART_VLS_index], [0,2]).reshape(num_jur,)
        #multiply F with phi for denominator
        
        K = np.sum(np.sum(data_array[:,risk,:,ART_VLS_index], axis=2)*scaling_factor_dropout[risk].reshape(10,1), axis=0)        
        # diagnosed and linked to care
        
        G = diagnosis_rate_risk[:,risk]*ltc_risk[:,risk]*np.sum((np.sum(data_array[:,risk,:,unaware_index],axis=2))*new_infections_data["testing_mult_fac_risk"][risk].reshape(5,1), axis=0)

        #entering care from unaware
        H = np.sum(gamma_t[risk].reshape(5,1)*np.sum(data_array[:,risk,:,aware_no_care_index], axis=2), axis=0)

        #total death art vls
        I = np.apply_over_axes(np.sum,death_per_month_risk_age_compartments[:,risk,:,ART_VLS_index],[0,2]).reshape(num_jur,)

        #number of art vls next time period
        J = (np.apply_over_axes(np.sum, data_array[:,risk,:,1:21], [1,2]).reshape(num_jur,) + np.sum(new_inf_per_month, axis=2)[:,risk] - np.apply_over_axes(np.sum, death_per_month_risk_age_compartments[:,risk,:,1:21],[1,2]).reshape(num_jur,)) * (aware_art_vls_prop[:,risk] + a_art_t[:,risk])
        
        dropout_rate_risk[:,risk] = (F+G+H-I-J)/K
        
#         if dropout_rate_risk[:,risk].any() < 0:
        dropout_rate_risk[:,risk][dropout_rate_risk[:,risk] < 0] = 0
        
    return dropout_rate_risk

def q_matrix(num_jur, new_infections_data, diagnosis_rate_risk, dropout_rate_risk, ltc_risk):
    
    Q_MAT = new_infections_data['q']
    Q_matrix = np.zeros((num_jur, number_of_risk_groups, number_of_compartments, number_of_compartments))
    
    for jur in range(num_jur):
        for risk in range(number_of_risk_groups):
            Q_mat = Q_MAT.copy()
            Q_mat[np.where(Q_mat == 12345)] = (1 - ltc_risk[jur,risk]) * diagnosis_rate_risk[jur,risk]*new_infections_data["testing_mult_fac_risk"][risk]
            Q_mat[np.where(Q_mat == 123456)] = dropout_rate_risk[jur,risk]
            Q_mat[np.where(Q_mat == 1234567)] = ltc_risk[jur,risk] * diagnosis_rate_risk[jur,risk]*new_infections_data["testing_mult_fac_risk"][risk]

            Q_matrix[jur,risk] = Q_mat
            
        
    return Q_matrix  

def q_mat_diag(Q_matrix, num_jur):
    
    Q_matrix_diagonal = np.zeros((num_jur, number_of_risk_groups, number_of_compartments, number_of_compartments))
    
    for jur in range(num_jur):
        for risk in range(number_of_risk_groups):
            Q_i = Q_matrix[jur,risk].copy()
            Q_i_sum = np.sum(Q_i, 1)
            Q_matrix_diagonal[jur,risk] = np.diag(Q_i_sum)
        
    return Q_matrix_diagonal

# pop_susceptible_12_years = data_array_cluster[:,:,0,:]

def aging(data_array, pop_susceptible_12_years):
    new_pop = np.zeros((num_jur, number_of_risk_groups, num_age, number_of_compartments))
    
    new_pop[:,:,1:,:] = data_array[:,:,0:num_age-1,:]
    new_pop[:,:,0,0] = pop_susceptible_12_years
    
    return new_pop

def extract_state(data_array, prep_values):

    state = []
    for i in range(num_jur): 
        data_array_cluster = data_array[[i],:,:,:]

        x = np.sum(data_array_cluster,axis=0)
        total_data_cluster_i = x[np.newaxis,:,:,:]
        #print(total_data_cluster_i.shape) #(1,3,88,22)
        
        # INFO: This line calculates and returns the proportion of population in total, prevalance and various care continuum stages 
        # FUNCTION NAME: calculate_proportions
        # FILE NAME: rate_prop_calculation_96.ipynb
        total_pop, prevalence_prop, unaware_prop, aware_no_art_prop, aware_art_vls_prop, _ = calculate_proportions(total_data_cluster_i, 1, number_of_risk_groups, unaware_index, aware_no_care_index, ART_VLS_index, VLS_index)

        prep_coverage = data_array_cluster[:,2,:,0]*prep_values[[i],2][:,np.newaxis]
        prep = np.round(np.apply_over_axes(np.sum, prep_coverage, [0,1]).item()/np.apply_over_axes(np.sum, data_array_cluster[:,2,:,0], [0,1]).item(), 4)
        prep_prop = np.array([0,0,prep])

        current_state = np.transpose(np.vstack((prevalence_prop, unaware_prop, aware_no_art_prop, aware_art_vls_prop, prep_prop)))
        state.append(current_state)
    
    return(state)

def initial_state(initial_data, prep_values):
    
    # INFO: This line extracts and returns the state of each jurisdiction in the simulation
    # FUNCTION NAME: extract_state
    # FILE NAME: rate_prop_calculation_96.ipynb
    state = extract_state(initial_data, prep_values)
    time = 0

    return initial_data, state, prep_values, time


