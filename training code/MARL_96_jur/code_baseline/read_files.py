import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from collections import defaultdict

age_begin = 13
age_end = 100
risk_groups = ['HM', 'HF', 'MSM']
num_risk = 3

#all_jur = 10
#age_groups = 88
number_of_risk_groups = 3
number_of_compartments = 22
dt = 1/12

#num_age = 88
#num_comp = number_of_compartments-2

#prep_efficiency = 0.99

#unaware_index = (1,5,9,13,17)
#aware_no_care_index = (2,6,10,14,18)
#ART_VLS_index = (3,4,7,8,11,12,15,16,19,20)
#VLS_index = (4,8,12,16,20)


#pop_growth_rate = 0

#gamma = np.array([[0.5,0.5,0.5,0.5,1],
#                  [0.5,0.5,0.5,0.5,1],
#                  [0.5,0.5,0.5,0.5,1]])

#scaling_factor_dropout = np.array([[1,1,1,1,1,1,1,1,0,0],
#                                   [1,1,1,1,1,1,1,1,0,0],
#                                   [1,1,1,1,1,1,1,1,0,0]])

def generate_age_mixing_mat(age_mat, age_begin, age_end):
    
    A1 = age_mat.values.reshape(8,8)    # age_mat- pandas dataframe, reshape(8,8) gives array of values excluding 1st row i.e. column name(age)
    #print(age_mat.values.shape)
    
    A = age_mat.columns.values         # gives (8,) array of column names (age)
    B = np.append(A, age_end+1)        # gives array of (9,1) with last element age_end+1
    
    
    B1 = B-age_begin                   # new array of B-age_begin (difference in age)
    #print(B, B1)
    
    #print(B1)
    
    C1 = np.empty([0,A.shape[0]])      #empty array of (0,8)
    D1 = np.zeros((A.shape[0]))        # array with zeros (8,) 1d array
    
    F1 =np.array([])
    for i in range(len(B1)-1):
        D1[i] = int(B1[i+1]-B1[i])
        F1 = np.append(F1, np.repeat(D1[i], D1[i]))
    #print(F1)
    D1 = D1.astype(int)
    
    for i in range(len(A1)):
        res1 = np.tile(A1[i],(D1[i],1))
        C1 = np.vstack((C1, res1))
    H1 = np.empty([0,88])
    
    for i in range(C1.shape[0]):
        res3 = np.array([])
        for j in range(C1.shape[1]):
            res3 = np.append( res3, (np.repeat(C1[i,j],D1[j])))
        H1 = np.vstack((H1, res3))
    
           
    G1 = np.zeros((88,88))
    for i in range(H1.shape[0]):
        G1[i] = H1[i]/F1
    
    age_mat = G1/100
    
    return(age_mat)

def read_new_inf_input(age_begin, age_end, risk_groups):
    
    path = os.getcwd()
    
    num_risk_groups = len(risk_groups)
    num_age_groups = age_end-age_begin+1
    
    excel = './input_files/input_new_infections.xlsx'
    
    
    """###"""
    q = pd.read_excel(excel, sheet_name='qmat').values.reshape(22,22)
    #print("\nNUMPY READ",q, q.shape, np.sum(q))
    
    """###"""
    Age_hm = pd.read_excel(excel, sheet_name='A_hm')
    
    Age_hf = pd.read_excel(excel, sheet_name='A_hf')
    
    Age_msm = pd.read_excel(excel, sheet_name='A_msm')
    
    
    A_hm = generate_age_mixing_mat(Age_hm, age_begin, age_end)
    A_hf = generate_age_mixing_mat(Age_hf, age_begin, age_end)
    A_msm = generate_age_mixing_mat(Age_msm, age_begin, age_end)
    
    age_mixing_final_mat = np.vstack((A_hm, A_hf, A_msm)).reshape(num_risk_groups,num_age_groups,num_age_groups)
    
    """###"""
    pi = pd.read_excel(excel, sheet_name='pi').values.reshape(20)
    
    """###"""
    num_sex_acts = pd.read_excel(excel, sheet_name='num_sex_acts') #female upper, female lower, male upper, male lower
    number_of_sex_acts_risk_age = np.zeros((num_risk_groups,num_age_groups))
    sex_act_calibration_param = [0.44, 90000000000000000, 90000000000000000]
    
    for risk in range(num_risk_groups):
        current_risk_group = risk_groups[risk]
        for age_index in range(num_age_groups):
            current_age = age_index+age_begin
    
            if((current_risk_group == 'HM') | (current_risk_group == "MSM") | (current_risk_group == "MSMIDU") | (current_risk_group == "IDUM")):
                upper = num_sex_acts[(num_sex_acts['Age_group'] == current_age)].Male_Upper.values
                lower = num_sex_acts[(num_sex_acts['Age_group'] == current_age)].Male_Lower.values
                
            elif((current_risk_group == 'HF') | (current_risk_group == "IDUF")):
                upper = num_sex_acts[(num_sex_acts['Age_group'] == current_age)].Female_Upper.values
                lower = num_sex_acts[(num_sex_acts['Age_group'] == current_age)].Female_Lower.values
            
            number_of_sex_acts_risk_age[risk,age_index] = lower+((upper-lower)/sex_act_calibration_param[risk])
    
    
    """###"""        
    condom_efficiency = pd.read_excel(excel, sheet_name='condom_efficiency').values
    
    """###"""
    prop_anal_acts = np.zeros((num_risk_groups, num_age_groups))
    prop_acts_pd = pd.read_excel(excel, sheet_name='prop_anal_acts')
    for risk in range(num_risk_groups):
        current_risk = risk_groups[risk]
        prop_anal_acts[risk] = prop_acts_pd[current_risk].values
    
    #print(prop_anal_acts, prop_anal_acts.shape)
    """###"""
    
    prop_casual_partner_v = pd.read_excel(excel, sheet_name='prop_casual_partner') 
    prop_casual_partner_risk = np.zeros((num_risk_groups, 2)) #columns [0]prob_casual	[1]prob_casual_only
    
    for risk in range(num_risk_groups):
        current_risk_group = risk_groups[risk]
        prop_casual_partner_risk[risk] = prop_casual_partner_v[(prop_casual_partner_v['Group'] == current_risk_group)].values[:,1:3]
    
    prop_casual_partner_risk_casual = prop_casual_partner_risk[:,0]
    prop_casual_partner_risk_casual_only = prop_casual_partner_risk[:,1]
    
    """###"""
    num_partner = pd.read_excel(excel, sheet_name='num_partner')
    num_partner_risk = np.zeros((num_risk_groups,2))
    
    for risk in range(num_risk_groups):
        current_risk_group = risk_groups[risk]
        num_partner_risk[risk] = num_partner[(num_partner['Group'] == current_risk_group)].values[:,1:3]
    
    num_partner_risk_casual_only = num_partner_risk[:,0]
    num_partner_risk_casual = num_partner_risk[:,1]
    
    #print(num_partner_risk_casual, num_partner_risk_casual_only)
    
    """###"""
    num_cas_part_main_cas = pd.read_excel(excel, sheet_name ='num_cas_part_main-cas').values
    
    """###"""
    prop_condom_use = pd.read_excel(excel, sheet_name='prop_condom_use')
    prop_condom_use_risk_age_casual = np.zeros((num_risk_groups, num_age_groups))
    prop_condom_use_risk_age_main = np.zeros((num_risk_groups, num_age_groups))
    
    
    prop_condom_use_risk_age_main[0] = prop_condom_use.values[:,3]
    prop_condom_use_risk_age_main[1] = prop_condom_use.values[:,1]
    prop_condom_use_risk_age_main[2] = prop_condom_use.values[:,5]
    
    prop_condom_use_risk_age_casual[0] = prop_condom_use.values[:,4]
    prop_condom_use_risk_age_casual[1] = prop_condom_use.values[:,2]
    prop_condom_use_risk_age_casual[2] = prop_condom_use.values[:,6]
    
    """###"""    
    trans_prob = pd.read_excel(excel, sheet_name='trans_prob')
    trans_prob_v_acts = np.zeros((num_risk_groups))
    trans_prob_a_acts = np.zeros((num_risk_groups))
    
    for risk in range(num_risk_groups):
        current_risk_group = risk_groups[risk]
        trans_prob_v_acts[risk] = trans_prob[(trans_prob['Group'] == current_risk_group)].vaginal.values
        trans_prob_a_acts[risk] = trans_prob[(trans_prob['Group'] == current_risk_group)].anal.values
        
    #print(trans_prob_v_acts, trans_prob_a_acts)
    
    """###""" 
    sex_mixing = pd.read_excel(excel, sheet_name='sex_mixing').values
    
    """###""" 
    #excel = 'input_estimating_unknown_rates_PATH.xlsx'
    excel = './input_files/input_estimating_unknown_rates_HOPE.xlsx'
    testing_mult_fac = pd.read_excel(excel, sheet_name = 'testing_mult_fac')
    testing_mult_fac_risk = np.zeros((num_risk_groups,5))
    
    for risk in range(num_risk_groups):
        current_risk_group = risk_groups[risk]
        testing_mult_fac_risk[risk] = testing_mult_fac[current_risk_group]
        
        
    """###"""
    excel = './input_files/input_new_infections.xlsx'
    age_mixing_diag = pd.read_excel(excel, sheet_name = 'age_mix_diagonals')
    #print(age_mixing_diag.values[0])
    #print(num_cas_part_main_cas)
    new_infections_data = {
        "q": q,
        "age_mixing_final_mat": age_mixing_final_mat,
        "pi": pi,
        "number_of_sex_acts_risk_age": number_of_sex_acts_risk_age,
        "condom_efficiency": condom_efficiency,
        "prop_anal_acts": prop_anal_acts,
        "prop_casual_partner_risk_casual": prop_casual_partner_risk_casual,
        "prop_casual_partner_risk_casual_only": prop_casual_partner_risk_casual_only,
        "num_partner_risk_casual": num_partner_risk_casual,
        "num_partner_risk_casual_only": num_partner_risk_casual_only,
        "num_cas_part_main_cas": num_cas_part_main_cas,
        "prop_condom_use_risk_age_main": prop_condom_use_risk_age_main,
        "prop_condom_use_risk_age_casual": prop_condom_use_risk_age_casual,
        "trans_prob_v_acts": trans_prob_v_acts,
        "trans_prob_a_acts": trans_prob_a_acts,
        "sex_mixing": sex_mixing,
        "testing_mult_fac_risk": testing_mult_fac_risk,
        "age_mixing_diagonals": age_mixing_diag}
    
    return(new_infections_data)

def read_death_rates(age_begin, age_end, risk_groups):
    
    num_risk_groups = len(risk_groups)
    num_age_groups = age_end-age_begin+1
    
    #excel = 'input_estimating_unknown_rates_PATH.xlsx'
    #excel = 'input_estimating_unknown_rates_HOPE.xlsx'
    excel = './input_files/input_estimating_unknown_rates_MOD_v2.xlsx'
    
    """death_rate_uninf = pd.read_excel(excel, sheet_name = 'death_prob_uninf')
    death_rate_inf = pd.read_excel(excel, sheet_name = 'death_prob_inf')
    death_rate_a200 = pd.read_excel(excel, sheet_name = 'death_prob_a200')
    death_rate_b200 = pd.read_excel(excel, sheet_name = 'death_prob_b200')
    
    death_rate_uninf_risk_age = np.zeros((num_risk_groups, num_age_groups))
    for risk in range(num_risk_groups):
        current_risk_group = risk_groups[risk]
        
        death_rate_uninf_risk_age[risk] = death_rate_uninf[current_risk_group].values
    
    death_rate_inf_v = death_rate_inf.values  
    death_rate_a200_age_2010 = death_rate_a200[2010].values
    death_rate_a200_age_2016 = death_rate_a200[2016].values
    death_rate_b200_age_2010 = death_rate_b200[2010].values
    death_rate_b200_age_2016 = death_rate_b200[2016].values
    
    
    death_rate_inf_no_ART_acute = 0
    death_rate_inf_no_ART_above_500 = 0
    death_rate_inf_no_ART_above_350_500 = 0
    death_rate_inf_no_ART_below_200 = 0
    death_rate_inf_ART_below_200_age = 0
    death_rate_inf_ART_200_350_age = 0
    death_rate_inf_ART_above_350 = 0
    """
    death_rate_a200_age_2010 = 0    
    death_rate_a200_age_2016 = 0
    death_rate_b200_age_2010 = 0
    death_rate_b200_age_2016 = 0
    death_rate_inf_v = 0
    
    #The following lines of code are used if I used MOD dataset
    death_rate_uninf = pd.read_excel(excel, sheet_name = 'death_prob_uninf')
    death_rate_inf_no_art = pd.read_excel(excel, sheet_name = 'death_prob_inf_no_art')
    death_prob_inf_art = pd.read_excel(excel, sheet_name = 'death_prob_inf_art')
    
    death_rate_uninf_risk_age = np.zeros((num_risk_groups, num_age_groups))
    for risk in range(num_risk_groups):
        current_risk_group = risk_groups[risk]
        
        death_rate_uninf_risk_age[risk] = death_rate_uninf[current_risk_group].values
    
    
    
    death_rate_inf_no_ART_acute = death_rate_inf_no_art[(death_rate_inf_no_art['CD4_category'] == "Acute")]["death_rate"].values
    death_rate_inf_no_ART_above_500 = death_rate_inf_no_art[(death_rate_inf_no_art['CD4_category'] == "CD4 >500")]["death_rate"].values
    death_rate_inf_no_ART_above_350_500 = death_rate_inf_no_art[(death_rate_inf_no_art['CD4_category'] == "CD4 350-500")]["death_rate"].values
    death_rate_inf_no_ART_above_200_350 = death_rate_inf_no_art[(death_rate_inf_no_art['CD4_category'] == "CD4 200-350")]["death_rate"].values
    death_rate_inf_no_ART_below_200 = death_rate_inf_no_art[(death_rate_inf_no_art['CD4_category'] == "CD4 <200")]["death_rate"].values
    
    death_rate_inf_ART_below_200_age = death_prob_inf_art["CD4_b_200"].values
    death_rate_inf_ART_200_350_age = death_prob_inf_art["CD4_200_350"].values
    death_rate_inf_ART_above_350_age = death_prob_inf_art["CD4_a_350"].values
    
    death_prob_data = {
        "death_rate_uninf_risk_age": death_rate_uninf_risk_age,
        "death_rate_inf": death_rate_inf_v,
        "death_rate_a200_age_2010": death_rate_a200_age_2010,
        "death_rate_a200_age_2016": death_rate_a200_age_2016,
        "death_rate_b200_age_2010": death_rate_b200_age_2010,
        "death_rate_b200_age_2016": death_rate_b200_age_2016,
        "death_rate_inf_no_ART_acute": death_rate_inf_no_ART_acute,
        "death_rate_inf_no_ART_above_500": death_rate_inf_no_ART_above_500,
        "death_rate_inf_no_ART_above_350_500": death_rate_inf_no_ART_above_350_500,
        "death_rate_inf_no_ART_above_200_350": death_rate_inf_no_ART_above_200_350,
        "death_rate_inf_no_ART_below_200": death_rate_inf_no_ART_below_200,
        "death_rate_inf_ART_below_200_age": death_rate_inf_ART_below_200_age,
        "death_rate_inf_ART_200_350_age": death_rate_inf_ART_200_350_age,
        "death_rate_inf_ART_above_350_age": death_rate_inf_ART_above_350_age}
    
    
    #print(death_prob_data)
    
    #print(death_prob_data["death_rate_uninf_risk_age"].shape, death_prob_data["death_rate_uninf_risk_age"])
    
    return(death_prob_data)

def calculate_deaths_vector(number_of_compartments, 
                            risk_groups, 
                            group, 
                            age, 
                            death_prob_data):
    
    current_age = age+13
    current_risk_group = risk_groups[group]
    #print(death_prob_data["death_rate_uninf_risk_age"].shape)
    """print("Group = ", current_risk_group)
    print("Age = ",current_age)
    print("year_to_simulate =", year_to_simulate)"""
    """ Need to multiply this with dt to get monthly rates if simulation time interval is monthly"""
    """
    
    death_col_annual = np.zeros((number_of_compartments,1))
    death_col_annual[0,:] = death_prob_data["death_rate_uninf_risk_age"][group,age].copy()
    
    #ART_more_200 = np.arange(1,18)
    #no_ART_less_200 = (17,18)
    #ART_less_200 = (19,20)
    
    ART_more_200 = np.array([3,4,7,8,11,12,15,16])
    no_ART = np.array([1,2,5,6,9,10,13,14,17,18])
    ART_less_200 = np.array([19,20])
    
    
    if(year_to_simulate < 2016):
        
        death_rate_after_200 = death_prob_data["death_rate_a200_age_2010"][age].copy()
        death_rate_before_200 = death_prob_data["death_rate_b200_age_2010"][age].copy()
    
    elif(year_to_simulate >= 2016):
        
        death_rate_after_200 = death_prob_data["death_rate_a200_age_2016"][age].copy()
        death_rate_before_200 = death_prob_data["death_rate_b200_age_2016"][age].copy()
    #print("death_prob_data",death_prob_data["death_rate_inf"][0,0].copy())
    
    death_col_annual[ART_more_200,:] = np.repeat(death_rate_after_200, len(ART_more_200)).reshape(len(ART_more_200),1)
    death_col_annual[no_ART,:] = np.repeat(death_prob_data["death_rate_inf"][0,0].copy(), len(no_ART)).reshape(len(no_ART),1)
    death_col_annual[ART_less_200,:] = np.repeat(death_rate_before_200, len(ART_less_200)).reshape(len(ART_less_200),1)        
    death_col_annual[21,:] = 0
    """
    
    # MODIFIED rates (include CD4 specific values)
    
    death_col_annual = np.zeros((number_of_compartments,1))
    death_col_annual[0,:] = death_prob_data["death_rate_uninf_risk_age"][group,age].copy()
    
    #print(death_prob_data["death_rate_inf_no_ART_acute"])
    death_col_annual[1,:] = death_prob_data["death_rate_inf_no_ART_acute"].copy()
    death_col_annual[2,:] = death_prob_data["death_rate_inf_no_ART_acute"].copy()
    death_col_annual[3,:] = death_prob_data["death_rate_inf_ART_above_350_age"][age].copy()
    death_col_annual[4,:] = death_prob_data["death_rate_inf_ART_above_350_age"][age].copy()
    
    death_col_annual[5,:] = death_prob_data["death_rate_inf_no_ART_above_500"].copy()
    death_col_annual[6,:] = death_prob_data["death_rate_inf_no_ART_above_500"].copy()
    death_col_annual[7,:] = death_prob_data["death_rate_inf_ART_above_350_age"][age].copy()
    death_col_annual[8,:] = death_prob_data["death_rate_inf_ART_above_350_age"][age].copy()  
    
    death_col_annual[9,:] = death_prob_data["death_rate_inf_no_ART_above_350_500"].copy()
    death_col_annual[10,:] = death_prob_data["death_rate_inf_no_ART_above_350_500"].copy()
    death_col_annual[11,:] = death_prob_data["death_rate_inf_ART_above_350_age"][age].copy()
    death_col_annual[12,:] = death_prob_data["death_rate_inf_ART_above_350_age"][age].copy()
    
    death_col_annual[13,:] = death_prob_data["death_rate_inf_no_ART_above_200_350"].copy()
    death_col_annual[14,:] = death_prob_data["death_rate_inf_no_ART_above_200_350"].copy()
    death_col_annual[15,:] = death_prob_data["death_rate_inf_ART_200_350_age"][age].copy()
    death_col_annual[16,:] = death_prob_data["death_rate_inf_ART_200_350_age"][age].copy()
    
    death_col_annual[17,:] = death_prob_data["death_rate_inf_no_ART_below_200"].copy()
    death_col_annual[18,:] = death_prob_data["death_rate_inf_no_ART_below_200"].copy()
    death_col_annual[19,:] = death_prob_data["death_rate_inf_ART_below_200_age"][age].copy()
    death_col_annual[20,:] = death_prob_data["death_rate_inf_ART_below_200_age"][age].copy()
    
    death_col_annual[21,:] = 0
        
    #print(1-death_col_annual)
    death_col_log = (-np.log(1-death_col_annual)) #*(1-0.56)#*(1-0.4543948)
    #print("group=",group,"age=",age,"\n")
    #print(np.sum(death_col_log))
    #print(death_col_log)
    """if(group == 0):
        if(age == 0):
            print(death_col_log)"""
    return(death_col_log)

def ltc_prep_values(ltc_excel, jur_list):
    
    df_hm_ltc = pd.read_excel(ltc_excel, sheet_name='jur_specific_care_cont_hm')
    df_hf_ltc = pd.read_excel(ltc_excel, sheet_name='jur_specific_care_cont_hf')
    df_msm_ltc = pd.read_excel(ltc_excel, sheet_name='jur_specific_care_cont_msm')
    
    ltc_risk = np.zeros((len(jur_list), number_of_risk_groups))
    
    for loc in range(len(jur_list)):
        d_hm = df_hm_ltc[df_hm_ltc['FIPS']==jur_list[loc]]
        d_hf = df_hf_ltc[df_hf_ltc['FIPS']==jur_list[loc]]
        d_msm = df_msm_ltc[df_msm_ltc['FIPS']==jur_list[loc]]

        ltc_vals = np.array([d_hm.LTC.values[0],
                             d_hf.LTC.values[0],
                             d_msm.LTC.values[0]])
        
        ltc_risk[loc] = ltc_vals
    
#     ltc_risk = np.array([df_hm_ltc[df_hm_ltc['FIPS']==loc_fips].LTC.values[0],
#                          df_hm_ltc[df_hf_ltc['FIPS']==loc_fips].LTC.values[0],
#                          df_hm_ltc[df_msm_ltc['FIPS']==loc_fips].LTC.values[0]])
    
#     prep_risk = np.array([0,
#                           0,
#                           (df_msm_ltc[df_msm_ltc['FIPS']==loc_fips].PrEP.values[0])/100])
    
    return ltc_risk

def M_x1_y1_value(new_infections_data):
    
    condom_awareness = np.array([0,0.53,0.53,0.53])
    prob_condom_efficency = new_infections_data["condom_efficiency"][0,0].copy()

    c = np.tile(condom_awareness,5).reshape(number_of_compartments-2)
    
    pi_v = new_infections_data["pi"].reshape((number_of_compartments-2)).copy()

    pi = pi_v[np.newaxis,:]
    p_v_x1 = new_infections_data["trans_prob_v_acts"].copy()[:,np.newaxis]
    p_a_x1 = new_infections_data["trans_prob_a_acts"].copy()[:,np.newaxis]

    p_bar_v_x1 = (1-prob_condom_efficency)*p_v_x1
    p_bar_a_x1 = (1-prob_condom_efficency)*p_a_x1
    
    num_sex_acts = new_infections_data["number_of_sex_acts_risk_age"].copy()
    prob_anal_acts = new_infections_data["prop_anal_acts"].copy()
    n_v_x1_y1 = num_sex_acts*(1-prob_anal_acts)
    n_a_x1_y1 = num_sex_acts*prob_anal_acts
    
    num_cas_part_main_cas = new_infections_data["num_cas_part_main_cas"].copy()[0:num_risk]
    nc = (num_cas_part_main_cas*2)/(num_sex_acts)
    nm = 1-nc
    
    prob_casual_only = new_infections_data["prop_casual_partner_risk_casual_only"].copy()[:,np.newaxis]
    prob_casual = new_infections_data["prop_casual_partner_risk_casual"].copy()[:,np.newaxis]

    prob_condom_casual = new_infections_data["prop_condom_use_risk_age_casual"].copy()
    prob_condom_main = new_infections_data["prop_condom_use_risk_age_main"].copy()

    prop_condom_use_calc = ((prob_casual_only*prob_condom_casual)+                                     #only casual
                                        (abs(prob_casual-prob_casual_only)*prob_condom_casual*nc)+ #casual among casual-main
                                        (abs(prob_casual-prob_casual_only)*prob_condom_main*nm)+     #main among casual-main
                                        ((1-prob_casual)*prob_condom_main)) 
    
    num_partners_tot = new_infections_data["num_partner_risk_casual"].copy() + \
                        new_infections_data["num_partner_risk_casual_only"].copy()
    
    lower1 = (1-(p_bar_v_x1*pi))[:,np.newaxis,:]

    upper1 = ((n_v_x1_y1*dt)[:,:,np.newaxis]*(((1-prop_condom_use_calc)[:,:,np.newaxis]*c[np.newaxis,np.newaxis,:])+
                                             prop_condom_use_calc[:,:,np.newaxis]))/(num_partners_tot*dt)[:,np.newaxis,np.newaxis]

    AA = lower1**upper1
    
    lower2 = (1-(p_v_x1*pi))[:,np.newaxis,:]

    upper2 = ((n_v_x1_y1*dt)[:,:,np.newaxis]*(1-prop_condom_use_calc)[:,:,np.newaxis]*(1-c[np.newaxis,np.newaxis,:]))/  \
                                            (num_partners_tot*dt)[:,np.newaxis,np.newaxis]

    BB = lower2**upper2
    
    lower3 = (1-(p_bar_a_x1*pi))[:,np.newaxis,:]

    upper3 = ((n_a_x1_y1*dt)[:,:,np.newaxis]*(((1-prop_condom_use_calc)[:,:,np.newaxis]*c[np.newaxis,np.newaxis,:])+
                                              prop_condom_use_calc[:,:,np.newaxis]))/(num_partners_tot*dt)[:,np.newaxis,np.newaxis]

    CC = lower3**upper3
    
    lower4 = (1-(p_a_x1*pi))[:,np.newaxis,:]

    upper4 = ((n_a_x1_y1*dt)[:,:,np.newaxis]*(1-prop_condom_use_calc)[:,:,np.newaxis]*(1-c[np.newaxis,np.newaxis,:]))/  \
                                            (num_partners_tot*dt)[:,np.newaxis,np.newaxis]

    DD = lower4**upper4
    
    M_x1_y1_i = 1-(1-(AA*BB*CC*DD))
    
    return M_x1_y1_i    


