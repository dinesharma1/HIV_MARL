import numpy as np
import warnings
warnings.filterwarnings("error",category=RuntimeWarning)

from choose_jurisdiction import *
from read_files import read_new_inf_input

#from ipynb.fs.full.choose_jurisdiction_96 import *
#from ipynb.fs.full.reading_files import read_new_inf_input

#age_begin = 13
#age_end = 100
risk_groups = ['HM', 'HF', 'MSM']
num_risk = 3

#all_jur = 10
#age_groups = 88
number_of_risk_groups = 3
#number_of_compartments = 22
dt = 1/12

#num_age = 88
#num_comp = number_of_compartments-2

#prep_efficiency = 0.99

unaware_index = (1,5,9,13,17)
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

all_jurisdictions = choose_jur()

num_jur = len(all_jurisdictions)

theta = 0.3
tau = 0.3
mu = 0.002
alpha = 0.8

c_r_n = 22.13
c_c_n = 10.36
n_n = 0.45
c_add = 52.66

c_r_p = 78.8
c_c_p = 58.91
n_r_p = 10.86
n_c_p = 5.88
c_cnf = 160.07

O_v0 = 16.59
W = 0.2 #[0.1,0.2,0.3]
# del_x = #outreached people

m_cl = 1000
m_nc = 1000
f_c = 56379
f_nc = 64851

m_o = 1000
f_o = 50000

R_art_0 = 235 #[117,235,300]
Y = 0.2 #[0.1,0.2,0.3]
m_r = 500
f_r = 22708 #[17977,22708,29330]

# p_art_0 = np.array([0.51041812, 0.5950631 , 0.57486548]) #need for all jur

def calculate_initial_proportion_values(data_array_cluster,new_inf_per_month, death_per_month_risk_age_compartments, new_infections_data):
    
    diagnosis_rate_risk = np.zeros((num_jur, number_of_risk_groups))

    data_array = data_array_cluster
    
    #a_unaware_t = np.round(a_unaware*dt, 6)
    #a_unaware_t = np.full((num_jur,num_risk),0.01*1/12)
    a_unaware_t = np.full((num_jur,num_risk),-0.001*1/12)

    # new infectiion per month
    A = np.sum(new_inf_per_month, axis=2)
    #print(A)

    # number of unaware population
    #B = p_unaware_0 * np.apply_over_axes(np.sum, data_array[:,:,:,1:21], [1,2])
    B = p_unaware_0 * np.apply_over_axes(np.sum, data_array_cluster[:,:,:,1:21],[2,3]).reshape(num_jur, num_risk)
    #print(B)

    # (current inf + new inf - total death) * (prop unaware + monthly change in proportion unaware)
    C = (np.apply_over_axes(np.sum, data_array[:,:,:,1:21], [2,3]).reshape(num_jur, num_risk) + A - np.apply_over_axes(np.sum, death_per_month_risk_age_compartments[:,:,:,1:21],[2,3]).reshape(num_jur,num_risk)) * (p_unaware_0+ a_unaware_t)
    #C = (np.apply_over_axes(np.sum, data_array[:,:,:,1:21], [2,3]).reshape(num_jur, num_risk) + A - np.apply_over_axes(np.sum, death_per_month_risk_age_compartments[:,:,:,1:21],[2,3]).reshape(num_jur,num_risk)) * (p_unaware_0) #+ a_unaware_t)
    #print(C) 

    # total deaths in each compartment (in unaware compartment?)
    D = np.apply_over_axes(np.sum, death_per_month_risk_age_compartments[:,:,:,unaware_index],[2,3]).reshape(num_jur,num_risk)
    #print(D)

    E1 = np.transpose(np.sum(data_array[:,:,:, unaware_index],axis=2))
    E2 = np.transpose(new_infections_data["testing_mult_fac_risk"])
    E2 = E2[:,:,np.newaxis]
    E = np.transpose(np.sum(E1 * E2, axis=0))
    #print("E",E)

    diagnosis_rate_risk = (A+B-C-D)/E
    #diagnosis_rate_risk[:,risk][diagnosis_rate_risk[:,risk] < 0] = 0

    #print("d_rate_risk",diagnosis_rate_risk)
    #raise Exception("Stopping for debugging")
    delta_1 = diagnosis_rate_risk

    """
    #INFO: I_t_0 = number of infected people at t = 0
    I_t_0 = np.apply_over_axes(np.sum, data_array_cluster[:,:,:,1:21],[2,3]).reshape(num_jur, num_risk)
    print(I_t_0) #(16,3)
    #raise Exception("Stopping for debugging")

    #INFO: i_1 = number of new infections at time t = 1
    i_1 = np.sum(new_inf_per_month, axis=2) #16,3

    #INFO: m_1 = number of deaths at time t = 1
    m_1 = np.apply_over_axes(np.sum, death_per_month_risk_age_compartments[:,:,:,1:21],[2,3]).reshape(num_jur, num_risk) #16,3
     
    #read p_unaware_0 from excel sheet 
    
    # delta1 calculated using 0.01 for a^hat_unaware
    a_hat_unaware = np.full((num_jur,num_risk),0.01)
    #print(a_hat_unaware)

    delta_1 = (-1 * ((I_t_0 + i_1 - m_1) * (p_unaware_0 + a_hat_unaware)) + (I_t_0 * p_unaware_0) + i_1) / (I_t_0 * p_unaware_0)
    print(delta_1)
    """
    
    #INFO: I_t_0 = number of infected people at t = 0
    I_t_0 = np.apply_over_axes(np.sum, data_array_cluster[:,:,:,1:21],[2,3]).reshape(num_jur, num_risk)

    N_t_0 = np.apply_over_axes(np.sum, data_array_cluster[:,:,:,0:21],[2,3]).reshape(num_jur, num_risk)

    x_t_a_0 = (delta_1*I_t_0*p_unaware_0 - mu*I_t_0*p_unaware_0) / (theta*shi_value(I_t_0,N_t_0))
    #print(x_t_a_0)
    return(x_t_a_0)

c_l = 54000

cd4_gt_350_index = (1,2,3,4,5,6,7,8,9,10,11,12)
cd4_200_350_index = (13,14,15,16)
cd_lt_200_index = (17,18,19,20)

QALY_val_gt_350 = 0.935
QALY_val_250_350 = 0.818
QALY_val_lt_200 = 0.702

def shi_value(I, N):
    return (0.0153/0.0057)*(I/N)

def cost(data_t_1, data_t, unaware_prop, aware_art_vls_prop, diagnosis_rate_risk, dropout_rate_risk, prep_rate, x_t_a_0):

    cost_risk = np.zeros((num_jur, number_of_risk_groups))

    for risk in range(len(risk_groups)):
        
        total_pop_t_risk = np.apply_over_axes(np.sum, data_t[:,risk,:,0:21],[1,2]).reshape(num_jur,)
        I_t_1 = np.apply_over_axes(np.sum, data_t_1[:,risk,:,1:21],[1,2]).reshape(num_jur,)
        p_t_1_unaware = unaware_prop[:,risk]
        p_t_1_art_vls = aware_art_vls_prop[:,risk]
        #print("ptartvls",p_t_1_art_vls)
        delta_t = diagnosis_rate_risk[:,risk]
        dropout_t = dropout_rate_risk[:,risk]
        r_t_a_risk = delta_t*I_t_1*p_t_1_unaware 
        shi = shi_value(I_t_1, total_pop_t_risk)
        x_t_a_risk = (r_t_a_risk - mu*I_t_1*p_t_1_unaware) / (theta*shi)
        n_t_a_risk = mu*(total_pop_t_risk - I_t_1*p_t_1_unaware - x_t_a_risk) + x_t_a_risk*theta*(1-shi) 
        X_v_risk = tau*c_r_n + (1-tau)*c_c_n + n_n + (1-alpha)*c_add
        Y_v_risk = tau*(c_r_p + n_r_p) + (1-tau)*(c_c_p + n_c_p) + c_cnf + (1-alpha)*c_add
        
        del_x = (x_t_a_risk - x_t_a_0[:,risk])/total_pop_t_risk
        
        O_v = O_v0*np.exp(del_x*W) #ERROR: CAUSING OVERFLOW VALUES

        """
        print("reached try catch 3")
        try:
            #print(O_v0) #16.59
            print("del_x",del_x)
            #print(W) #0.2
            O_v = O_v0*np.exp(del_x*W)
        except Exception as error:
            print("Something went wrong 3")
            print('An exception occurred: {}'.format(error))
            exit(0)
        #"""
        
        X_f_cl_a = ((r_t_a_risk + n_t_a_risk)*alpha / m_cl)*f_c
        X_f_ncl_a = ((r_t_a_risk + n_t_a_risk)*(1 - alpha) / m_nc)*f_nc
        X_f_o_a = (x_t_a_risk / m_o)*f_o

        cost_of_testing = r_t_a_risk*Y_v_risk + x_t_a_risk* O_v + n_t_a_risk*X_v_risk + X_f_cl_a + X_f_ncl_a + X_f_o_a

        d_t_a_risk = (1 - dropout_t)*I_t_1*p_t_1_art_vls
        del_p_art = p_t_1_art_vls - p_art_0[:,risk]
        R_v_risk = R_art_0*np.exp(del_p_art*Y) #ERROR: CAUSING OVERFLOW VALUES

        """
        print("reached try catch 2")
        try:
            #print(del_p_art.shape) #(50,)
            #print(Y) #0.2
            #print(R_art_0) #235
            del_p_art = np.round(del_p_art,2)
            print(del_p_art)
            R_v_risk = np.round(R_art_0*np.exp(del_p_art*Y),2)
            print("rvrisk",R_v_risk) #values becoming inf here why
            print("Worked 2")
            #exit(0)
        except Exception as error:
            print("Something went wrong 2")
            print('An exception occurred: {}'.format(error))
            print("pt1artvls",p_t_1_art_vls)
            print(p_art_0[:,risk])
            del_p_art = np.round(del_p_art)
            print("del_p_art",del_p_art)
            #R_v_risk = np.round(R_art_0*np.exp(del_p_art*Y),2)
            #print("rvrisk",R_v_risk) #values becoming inf here why
            exit(0)
        #"""

        E_f_a = (d_t_a_risk / m_r)*f_r

        cost_retention_in_care = d_t_a_risk*R_v_risk + E_f_a

        if risk == 2: #msm

            prep_adherence_per_person_per_year = 1431
            prep_medication_per_person_per_year = 12599

            prep_cost = np.sum(data_t_1[:,risk,:,0], axis=1)*prep_rate[:,risk]* \
                            (prep_adherence_per_person_per_year + prep_medication_per_person_per_year)

        else:
            prep_cost = 0

        cost_risk[:,risk] = cost_of_testing + cost_retention_in_care + prep_cost
        
    return dt*cost_risk

def benefit(data_array):
    L_t_risk = np.zeros((num_jur, number_of_risk_groups))

    for risk in range(len(risk_groups)):

        num_uninfected = np.sum(data_array[:,risk,:,0], axis=1)
        num_over_350 = np.apply_over_axes(np.sum, data_array[:,risk,:,cd4_gt_350_index], [0,2]).reshape(num_jur,)
        num_250_350 = np.apply_over_axes(np.sum, data_array[:,risk,:,cd4_200_350_index], [0,2]).reshape(num_jur,)
        num_below_250 = np.apply_over_axes(np.sum, data_array[:,risk,:,cd_lt_200_index], [0,2]).reshape(num_jur,)

        benefit_risk = 1*num_uninfected + QALY_val_gt_350*num_over_350 + QALY_val_250_350*num_250_350 + QALY_val_lt_200*num_below_250

        L_t_risk[:,risk] = benefit_risk
        
    return c_l*dt*L_t_risk
