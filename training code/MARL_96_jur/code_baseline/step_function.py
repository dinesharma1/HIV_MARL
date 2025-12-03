import numpy as np

from read_files import *
from rate_prop_calc import *
from cost_benefit import *
from choose_jurisdiction import *

age_begin = 13
age_end = 100
risk_groups = ['HM', 'HF', 'MSM']
#num_risk = 3

#all_jur = 10
age_groups = 88
number_of_risk_groups = 3
number_of_compartments = 22
dt = 1/12

#num_age = 88
#num_comp = number_of_compartments-2

#prep_efficiency = 0.99

unaware_index = (1,5,9,13,17)
aware_no_care_index = (2,6,10,14,18)
ART_VLS_index = (3,4,7,8,11,12,15,16,19,20)
VLS_index = (4,8,12,16,20)


pop_growth_rate = 0

gamma = np.array([[0.5,0.5,0.5,0.5,1],
                  [0.5,0.5,0.5,0.5,1],
                  [0.5,0.5,0.5,0.5,1]])

#scaling_factor_dropout = np.array([[1,1,1,1,1,1,1,1,0,0],
#                                   [1,1,1,1,1,1,1,1,0,0],
#                                   [1,1,1,1,1,1,1,1,0,0]])

all_jurisdictions = choose_jur()

num_jur = len(all_jurisdictions)

ltc_excel ='./input_files/CareContinuum-by_jur 7-26-2021 HT V2.xlsx'

ltc_risk = ltc_prep_values(ltc_excel, all_jurisdictions)  # linked to care rate

new_infections_data = read_new_inf_input(age_begin, age_end, risk_groups) # new infection data from excel

M_x1_y1_i = M_x1_y1_value(new_infections_data)

death_prob_data = read_death_rates(age_begin, age_end, risk_groups)

death_rate_risk_age_compartments = np.zeros((num_jur, number_of_risk_groups, age_groups, number_of_compartments))

for risk in range(len(risk_groups)):
    for age in range(age_groups):
        death_rate_risk_age_compartments[:,risk,age,] = calculate_deaths_vector(number_of_compartments, risk_groups, risk, age, death_prob_data).reshape(22)

def get_action(min_, max_, val):
    new = min_ + (max_ - min_) * (val + 1)/ 2 
    return new

def change_action_range(action):
    for i in range(3):
        if action[i] < -1:
            action[i] = -1
            
        elif action[i] > 1:
            action[i] = 1
            
        action[i] = get_action(-0.01, 0, action[i])
        
    for i in range(3,6):
        if action[i] < -1:
            action[i] = -1
            
        elif action[i] > 1:
            action[i] = 1
            
        action[i] = get_action(0, 0.08, action[i])
        
    for i in range(6,8):
        if action[i] < -1:
            action[i] = -1
            
        elif action[i] > 1:
            action[i] = 1
            
        action[i] = get_action(0,0, action[i])
        
    if action[8] < -1:
        action[8] = -1
        
    elif action[8] > 1:
        action[8] = 1
        
    action[8] = get_action(0,0.08, action[8])
    
    action = action.reshape(3,3)
    
    return action

def step(data_array, action, prep_values, current_time, x_t_a_0):

    #print("current_time",current_time)
    #raise Exception("Stopping for debugging")
    
    a_unaware = np.zeros((num_jur, 3))
    a_art = np.zeros((num_jur, 3))
    a_prep = np.zeros((num_jur, 3))
    for i in range(num_jur):
        action[i] = change_action_range(action[i]) #(3,3)
        #a_unaware[[i],:] = action[i][0]
        a_unaware[[i],:] = action[i][0]
        a_art[[i],:] = action[i][1]
        a_prep[[i],:] = action[i][2]
        #print(action[i].shape)
    #print(np.array(a_unaware).shape) #(8,3)
    #print(a_unaware)
    #raise Exception("Stopping for debugging")

    #prep
    prep_rate = prep_values + a_prep    
    #print(prep_rate.shape)
    #raise Exception("Stopping for debugging")
    
    pop_susceptible_12_years = data_array[:,:,0,0]
    
    total_reward = 0
    total_inf = 0
    total_cost = 0
    done = False

    # FUNCTION NAME: calculate_proportions
    # FILE NAME: rate_prop_calculation_96.ipynb
    total_pop, prevalence_prop, unaware_prop, aware_no_art_prop, aware_art_vls_prop,_ = \
        calculate_proportions(data_array, num_jur, number_of_risk_groups, unaware_index, aware_no_care_index, ART_VLS_index, VLS_index)

    new_inf_per_month = new_infections_per_month(num_jur, data_array, new_infections_data, M_x1_y1_i, prep_rate)
    #print("new_inf_per_month",new_inf_per_month)
    death_per_month_risk_age_compartments = data_array*death_rate_risk_age_compartments*dt
    #print("death_per_month_risk_age_comp",death_per_month_risk_age_compartments)

    # FUNCTION NAME: diagnosis_rate
    # FILE NAME: rate_prop_calculation_96.ipynb
    diagnosis_rate_risk = diagnosis_rate(data_array, num_jur, a_unaware, unaware_index, number_of_risk_groups, new_inf_per_month, unaware_prop, death_per_month_risk_age_compartments)

    dropout_rate_risk = dropout_rate(num_jur, a_art, ART_VLS_index, diagnosis_rate_risk, ltc_risk, gamma, number_of_risk_groups, data_array, new_inf_per_month, unaware_prop, aware_no_art_prop, aware_art_vls_prop, death_per_month_risk_age_compartments)

    Q_matrix = q_matrix(num_jur, new_infections_data, diagnosis_rate_risk, dropout_rate_risk, ltc_risk)

    Q_matrix_diagonal = q_mat_diag(Q_matrix, num_jur)

    #environment step is happening here - i think -ss
    for i in range(12):

        #print("i",i)

        new_data = np.zeros((num_jur, number_of_risk_groups, age_groups, number_of_compartments))

        data_t_1 = data_array.copy()

        # For each risk group:
        for risk in range(number_of_risk_groups):

            #calculate flow of infected to diff compartments and subtract from that compartment
            new_data[:,risk,:,:] = data_array[:,risk,:,:] + \
                                    np.matmul(data_array[:,risk,:,:], Q_matrix[:,risk,:,:]) - \
                                    np.matmul(data_array[:,risk,:,:], Q_matrix_diagonal[:,risk,:,:]) - \
                                    death_per_month_risk_age_compartments[:,risk,:,:]

            #subtract from susceptible and add to acute unaware
            new_data[:,risk,:,0] = new_data[:,risk,:,0] - new_inf_per_month[:,risk,:]
            
            new_data[:,risk,:,1] = new_data[:,risk,:,1] + new_inf_per_month[:,risk,:]

            #add the total deaths to last column
            new_data[:,risk,:,21] = np.sum(death_per_month_risk_age_compartments[:,risk,:,:], axis=2)

        # FUNCTION NAME: cost
        # FILE NAME: cost_benefit_96.ipynb
        cost_per_month = cost(data_t_1, new_data, unaware_prop, aware_art_vls_prop, diagnosis_rate_risk, dropout_rate_risk, prep_rate, x_t_a_0)

        benefit_per_month = benefit(new_data)

        reward_per_month = benefit_per_month - cost_per_month

        total_reward += reward_per_month
        
        total_cost += cost_per_month
        
        total_inf += new_inf_per_month

        data_array = new_data.copy()
        
    new_pop_dist = aging(data_array, pop_susceptible_12_years*(1+pop_growth_rate)) # adding new pop
    
    new_state = extract_state(new_pop_dist, prep_rate)                        
    #next_state = (new_pop_dist,new_state1,new_state2,new_state3,new_state4,new_state5,new_state6,new_state7,new_state8, prep_rate, current_time+1)

    reward_cluster = []
    inf_cluster = []
    total_cost_list = []
    reward_list = []
    total_new_inf_list = []
    for i in range(num_jur):
        reward_cluster.append(total_reward[[i],:])
        inf_cluster.append(total_inf[[i],:])
        total_cost_list.append(total_cost[[i],:])
        reward_list.append(-np.sum(inf_cluster[i]))
        total_new_inf_list.append(np.sum(inf_cluster[i]))
    #print(reward_list)
    #print("reached after reward")
    #raise Exception("Stopping for debugging")

    penalty_list = []
    for i in range(num_jur):
        penalty_list.append(np.sum(total_cost_list[i]) - budget_values[i])
        if (np.sum(total_cost_list[i]) > budget_values[i]):
            reward_list[i] -= (np.sum(total_cost_list[i])-budget_values[i])
    #raise Exception("Stopping for debugging")

    if current_time+1 == 12:
        done = True
    
    #print("reached after cost")
    
    return new_pop_dist, new_state, prep_rate, current_time+1, reward_list, total_new_inf_list, penalty_list, done



