#!/usr/bin/env python3
import os
from baselines.common import tf_util as U
from baselines.common.tf_util import load_variables
import gymfc
import numpy as np
import gym
from mpi4py import MPI
from plot_step_response import plot_step_response
from train_agent import train
from os import system
import time
# from disturbance import wind
# os.environ['OPENAI_LOGDIR'] = '/home/p1/gymfc/gymfc/Logs'
# os.environ['OPENAI_LOG_FORMAT'] = 'stdout,tensorboard'

def main():
    # seed = 13
    num_timesteps = 1e7
    train_model = False
    env_id = 'AttFC_GyroErr-MotorVel_M4_Ep-v0'
    
    if train_model:
        for seed in range(1,100,1):
            print('-------------Initiate Learning----------------')
            conf_dir='/home/p1/gymfc/gymfc/Logs/seed_'+str(seed)
            from baselines import logger
            logger.configure(dir=conf_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])
            # train the model
            train(num_timesteps=num_timesteps, train_model=train_model, seed=seed, env_id=env_id)
    else:
        seed = 34
        model_path = '/home/p1/gymfc/gymfc/models/test/PPO_seed_38_rew_6.8'

        # construct the model object, load pre-trained model and render

        # if a KeyError: 'pi##/obfilter/runningsum:0' error occurs, \
        # add a pi_name parameter to the train function call with the trained agent's seed num 
     
        pi = train(num_timesteps=1, seed=seed, train_model=train_model, env_id=env_id, pi_name=34)
        U.load_variables(model_path)

        env = gym.make(env_id)
        
        # env.render()

        best_err = np.inf
        worst_err = 0

        tot_sim_num = 5

        # disturb = False

        for k in range(1,tot_sim_num+1):  # test for 10 runs, plot best run

            ob = env.reset()
            actuals = []
            desireds = []
            print('Begin Sim num: {}'.format(k))
            time.sleep(4)

            while True:
                desired = env.omega_target
                actual = env.omega_actual
                actuals.append(actual)
                desireds.append(desired)
                # print ("Desired rate=", desired, " Actual rate=", actual)
                action = pi.act(stochastic=False, ob=ob)[0]
                # if disturb:
                #     action=wind(action, info[0])
                ob, _, done, info =  env.step(action)
                print(info)
                if done:
                    break

            if env_id == 'AttFC_GyroErr-MotorVel_M4_Ep-v0':    # test episodal enviorment
                err_num = sum(np.abs(sum(np.array(desireds)-np.array(actuals))))
                err_den = sum(np.abs(np.array(desireds[0])-np.array(actuals[0])))*len(desireds)
                err = err_num/err_den
                print ('Norm error =', err)
                if err < best_err:
                    print ('New best run')
                    best_err = err
                    best_des = desireds
                    best_act = actuals
                if err > worst_err:
                    print ('New worst run')
                    worst_err = err
                    worst_des = desireds
                    worst_act = actuals

            if env_id == 'AttFC_GyroErr-MotorVel_M4_Con-v0':    # test continous enviorment
                np.savetxt('des_seed_{}_run{}.csv'.format(seed,_), np.array(best_des), delimiter=',')
                np.savetxt('act_seed_{}_run{}.csv'.format(seed,_), np.array(best_act), delimiter=',')  

        if env_id == 'AttFC_GyroErr-MotorVel_M4_Ep-v0':
            plot_step_response(desired=np.array(best_des), actual=np.array(best_act), title='best', seed=seed)
            plot_step_response(desired=np.array(worst_des), actual=np.array(worst_act), title='worst', seed=seed)       
            

if __name__ == '__main__':
    main()