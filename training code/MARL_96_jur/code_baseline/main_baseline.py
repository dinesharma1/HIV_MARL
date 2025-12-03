import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import torch

#import custom modules
from choose_jurisdiction import *
from step_function import *
from PPO_baseline import *
from read_files import read_new_inf_input
from read_files import  M_x1_y1_value
from cost_benefit import calculate_initial_proportion_values

# set device to cpu or cuda
device = torch.device('cpu')

#"""
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))

    gpu_t = torch.cuda.get_device_properties(0).total_memory 
    print("Total memory:", gpu_t/1024/1024/1024) #in GB #8GB
    
    gpu_r = torch.cuda.memory_reserved(0)
    print("Reserved memory:", gpu_r/1024/1024) #in MB #38MB

    gpu_a = torch.cuda.memory_allocated(0)
    print("Allocated memory:",gpu_a/1024/1024) #in MB #27.75MB
    
else:
    print("Device set to : cpu")
#"""
    
print("============================================================================================")


################################### Training ###################################

#set num_jurisdictions
#instead use num_jur defined in choose_jurisdiction.ipynb

####### initialize environment hyperparameters ######

action_std_decay_rate = 0.0046
min_action_std = 0.05
action_std_decay_freq = 1000

has_continuous_action_space = True

max_ep_len = 12                   # max timesteps in one episode
max_training_timesteps = 10000 #50000 #100000   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 10     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
save_model_freq = 2000      # save model frequency (in num timesteps)
plot_freq = 1200

action_std = 0.4

#####################################################


## Note : print/log frequencies should be > than max_ep_len


################ PPO hyperparameters ################


update_timestep = 120     # update policy every n timesteps
K_epochs = 20               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma_ = 0.99                # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.0003       # learning rate for critic network

random_seed = 11   # set random seed if required (0 = no random seed)

#####################################################

env_name = 'HIV Jurisdiction'

print("training environment name : " + env_name)

# env = gym.make(env_name)

# state space dimension
obs_dim = 15
state_dim = 120

# action space dimension
if has_continuous_action_space:
    action_dim = 9
else:
    action_dim = 1

############# print all hyperparameters #############

print("--------------------------------------------------------------------------------------------")

print("max training timesteps : ", max_training_timesteps)
print("max timesteps per episode : ", max_ep_len)

print("model saving frequency : " + str(save_model_freq) + " timesteps")
print("log frequency : " + str(log_freq) + " timesteps")
print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

print("--------------------------------------------------------------------------------------------")

print("observation space dimension : ", obs_dim)
print("state space dimension : ", state_dim)
print("action space dimension : ", action_dim)

print("--------------------------------------------------------------------------------------------")

if has_continuous_action_space:
    print("Initializing a continuous action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("starting std of action distribution : ", action_std)
    print("decay rate of std of action distribution : ", action_std_decay_rate)
    print("minimum std of action distribution : ", min_action_std)
    print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")

else:
    print("Initializing a discrete action space policy")

print("--------------------------------------------------------------------------------------------")

print("PPO update frequency : " + str(update_timestep) + " timesteps") 
print("PPO K epochs : ", K_epochs)
print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma_) : ", gamma_)

print("--------------------------------------------------------------------------------------------")

print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate critic : ", lr_critic)

if random_seed:
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    #env.seed(random_seed) 
    np.random.seed(random_seed)

#####################################################

print("Number of Jurisdictions being modeled:",num_jur)

#directory path code
# if no directory name provided use current date time
dir_name = "Exp_" + str(datetime.now()).split(".")[0].replace(" ", "_").replace(":","-")

#if using windows, save path
cwd = os.getcwd()
cwd = cwd + r"\results_baseline" 
save_path = os.path.join(cwd, dir_name)
#cwd = r'%s' % cwd
#os.mkdir(cwd)
os.makedirs(save_path)

print("============================================================================================")

################# training procedure ################

# initialize a PPO agent
ppo_agent = []
for i in range(num_jur):
    ppo_agent.append(PPO(obs_dim, state_dim, action_dim, lr_actor, lr_critic, gamma_, K_epochs, eps_clip, has_continuous_action_space, action_std))

# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print("============================================================================================")


# logging file
"""
log_f = open(log_f_name,"w+")
log_f.write('episode,timestep,reward\n')
"""

rew_list = [[] for x in range(num_jur)] 

# printing and logging variables
print_running_reward = np.zeros(num_jur) 

print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0

#calculate_initial_proportion_values(data_array_cluster)
# ESS_SOC_max = 1500
# T = 24
# training loop
while time_step <= max_training_timesteps:

    # INFO: This code block gets the initial data, states, prep values, and simulation time from the excel data file.
    # FUNCTION NAME: initial_state
    # FILE NAME: rate_prop_calculation_96.ipynb
    # NOTES: data_array_cluster and prep_values initialized in the called file; no hardcoding
    # NOTES: state is a tuple of 10 values -> 0 (inital data), 1-8 (states), 9 (prepvalues), 10(time)
    initial_data_ss, state, prep_values_ss, time_ss = initial_state(data_array_cluster,prep_values) 
    
    current_ep_reward = np.zeros(num_jur) 

    # INFO: this code block initializes the variables before the start of the t loop.
    data_in_t_loop = initial_data_ss
    #state                                  #already initialized above
    prep_values_in_t_loop = prep_values_ss
    time_in_t_loop = time_ss

    # INFO: This code reads excel file and assigns values related to new infections behavior
    # FUNCTION NAME: read_new_inf_input
    # FILE NAME: reading_files.ipynb
    new_infections_data = read_new_inf_input(age_begin, age_end, risk_groups)

    # INFO: This calculates something??? I think it reads transmission variables like num sex acts, condom efficacy, etc.
    # (Required to calculate new infections per month)
    # FUNCTION NAME: M_x1_y1_value
    # FILE NAME: reading_files.ipynb
    M_x1_y1_i = M_x1_y1_value(new_infections_data)

    # INFO: This code block gets the initial data, states, prep values, and simulation time from the excel data file.
    # FUNCTION NAME: new_infections_per_month
    # FILE NAME: rate_prop_calculation_96.ipynb
    #using a_prep:
    """
    #prep_rate = prep_values_ss + a_prep
    #new_inf_per_month_0 = new_infections_per_month(num_jur, initial_data_ss, new_infections_data, M_x1_y1_i, prep_rate)
    """
    #ignoring a_prep:
    new_inf_per_month_0 = new_infections_per_month(num_jur, initial_data_ss, new_infections_data, M_x1_y1_i, prep_values_ss)
    #print(np.sum(new_inf_per_month_0,axis=2))
    
    # INFO: This code reads excel file and assigns values related to death rate
    # FUNCTION NAME: read_death_rates
    # FILE NAME: reading_files.ipynb
    death_prob_data = read_death_rates(age_begin, age_end, risk_groups)

    # INFO: This code block stratifies death rate by risk, age and compartment
    # FUNCTION NAME: calculate_deaths_vector
    # FILE NAME: reading_files.ipynb
    death_rate_risk_age_compartments = np.zeros((num_jur, number_of_risk_groups, age_groups, number_of_compartments))

    for risk in range(len(risk_groups)):
        for age in range(age_groups):
            death_rate_risk_age_compartments[:,risk,age,] = calculate_deaths_vector(number_of_compartments, risk_groups, risk, age, death_prob_data).reshape(22)
    death_per_month_risk_age_compartments_0 = initial_data_ss * death_rate_risk_age_compartments * dt
    #print(death_per_month_risk_age_compartments_0)

    # INFO: This function call calculates the initial population, proportion unaware, proportion aware at time zero and diagnosis rate and time 1
    # FUNCTION NAME: calculate_initial_proportion_values
    # FILE NAME: cost_benefit_96.ipynb
    x_t_a_0 = calculate_initial_proportion_values(initial_data_ss, new_inf_per_month_0, death_per_month_risk_age_compartments_0, new_infections_data)
    #raise Exception("Stopping for debugging")
       
    # INFO: We run the following code for each timestep of the simulation.
    for t in range(0, max_ep_len+1):
        #print("t:",t)
        
        full_state = np.vstack(state).flatten() 
        #print(np.shape(full_state)) shape = 15 care cont states x num of jurisdictions

        # INFO: This line selects the action using the state of each jurisdiction from the policy
        # FUNCTION NAME: select_action
        # FILE NAME: PPO_algo_96.ipynb
        action = [ppo_agent[i].select_action(state[i].flatten()) for i in range(num_jur)]

        # INFO: This line calls the simulation step function, and returns the rewards for each jurisdiction
        # FUNCTION NAME: step
        # FILE NAME: step_function_96.ipynb
        data_in_t_loop, state, prep_values_in_t_loop, time_in_t_loop, reward, new_inf, penalty, done = step(data_in_t_loop, action, prep_values_in_t_loop, time_in_t_loop, x_t_a_0)
        #print("t:",t)
        

        # INFO: This loop calls saves the reward and is_terminals in the buffer
        for i in range(num_jur):
            ppo_agent[i].buffer.full_states.append(torch.FloatTensor(full_state).to(device))
            ppo_agent[i].buffer.rewards.append(reward[i])
            ppo_agent[i].buffer.is_terminals.append(done)
            current_ep_reward[i] += reward[i]
            ppo_agent[i].buffer.new_inf.append(new_inf[i])
            ppo_agent[i].buffer.penalty.append(penalty[i])

        
        time_step +=1
        #print("time_step:",time_step)
        #ltc_risk += ltc_increment

        if time_step % print_freq == 0:
            new_inf_list = []
            penalty_list = []
            for i in range(num_jur):
                new_inf_list.append(ppo_agent[i].buffer.new_inf)
                penalty_list.append(ppo_agent[i].buffer.penalty)
                #print("jurisdiction:",i)
                #print(np.array((ppo_agent[i].buffer.new_inf)).shape)
            np.savetxt("new_inf_baseline.csv", new_inf_list, delimiter=",")
            np.savetxt("penalty_baseline.csv", penalty_list, delimiter=",")
            #exit(0)
        
        # INFO: This code block updates the policy learned for each PPO agent
        # FUNCTION NAME: update
        # FILE NAME: PPO_algo_96.ipynb
        if time_step % update_timestep == 0:
            update_start = datetime.now()
            for i in range(num_jur):
                ppo_agent[i].update()
            print("Update Time Taken  : ", datetime.now() - update_start)

        # INFO: if continuous action space; then decay action std of ouput action distribution
        if has_continuous_action_space and time_step % action_std_decay_freq == 0:
            for i in range(num_jur):
                ppo_agent[i].decay_action_std(action_std_decay_rate, min_action_std)

        # INFO: log in logging file - not really used? can remove??? check.
        """
        if time_step % log_freq == 0:

            # log average reward till last episode
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)

            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f.flush()

            log_running_reward = 0
            log_running_episodes = 0
        """

        # INFO: printing average reward
        if time_step % print_freq == 0:

            # print average reward till last episode
            print_avg_reward = np.zeros(num_jur)
            for i in range(num_jur):
                print_avg_reward[i] = round((print_running_reward[i] / print_running_episodes),2)
                print("Agent{} => Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i+1, i_episode, time_step, print_avg_reward[i]))
                rew_list[i].append(print_avg_reward[i])

            print_running_reward = np.zeros(num_jur) #Sonza changes
            print_running_episodes = 0

            # INFO: save rewards as a csv file
            np.savetxt("avg_rew_baseline.csv", rew_list, delimiter=",")

        # INFO: Display and save result figures here    
        if time_step % plot_freq == 0:
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")

            x = 12 #2 #num_jur/4 #hardcoded
            y = 8 #hardcoded
            fig,ax = plt.subplots(x,y,sharex=False, sharey=False, figsize=(40,28)) #(20,14)
            fig.tight_layout(h_pad=3, w_pad=1)
            
            i = 0
            for j in range(x): #hardcoded
                for k in range(y): #hardcoded
                    if(i==num_jur):
                        break
                    ax[j][k].plot(range(len(rew_list[i])), rew_list[i])
                    #ax[j][k].set_ylim(-1e9,0.01)
                    ax[j][k].set_title("Rewards for Jurisdiction {}".format(all_jurisdictions[i]),pad=12)
                    i += 1

            # fig.suptitle('Rewards for all the agents')
            plt.savefig(os.path.join(save_path, str(time_step)))
            #plt.show()
            plt.close()
                        
        # INFO: save model weights - where is it being saved + how to load??? check.
        """
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model")

            for i in range(num_jur):
                ppo_agent[i].save(checkpoint_path[i])

            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
        """
            
        # break; if the episode is over
        #print("done?",done)
        if done:
            break
    #end of t loop
    #timestep currently = t + (12-1) ie (max ep len - 1)
      
    for i in range(num_jur):
        print_running_reward[i] += current_ep_reward[i]

    print_running_episodes += 1

    #log is only logging the first jurisdiction - do we need this??? check.
    log_running_reward += current_ep_reward[0] 
    log_running_episodes += 1

    i_episode += 1

# INFO: save rewards as a csv file
#np.savetxt("rew_baseline.csv", rew_list, delimiter=",")

#log_f.close()
# env.close()


