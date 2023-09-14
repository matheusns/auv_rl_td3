#!/usr/bin/env python3.8
import os
import rospy
import gymnasium as gym
from stable_baselines3 import TD3
from auv_rl_gym.task_envs.desistek_saga.auto_docking import AutoDocking

class TD3Trainer:
    def __init__(self, timesteps, log_dir):
        self.timesteps = timesteps
        self.env = gym.make('DesistekSagaAutoDocking-v0')
        self.model = TD3("MlpPolicy", 
                        self.env, verbose=2,
                        train_freq=(1,"episode"),
                        tensorboard_log=log_dir,
                        device="auto")

    def learn(self):
        self.model.learn(total_timesteps=self.timesteps)
    def save(self, path):
        self.model.save(path)
        vec_env = self.model.get_env()

if __name__ == "__main__":
    rospy.init_node('env_checker', anonymous=True)
    
    timesteps = rospy.get_param("/desistek_saga/learn/timesteps")
    log_dir = rospy.get_param("/desistek_saga/log/dir")
    save_dir = rospy.get_param("/desistek_saga/log/save")
    model_name = rospy.get_param("/desistek_saga/log/model_name")
    
    td3 = TD3Trainer(timesteps, log_dir)
    initial_time = rospy.Time.now()
    td3.learn()
    td3.save(os.path.join(save_dir, model_name))
    elapsed_time = rospy.Time.now() - initial_time
    rospy.loginfo("Elapsed time: %s", elapsed_time)
    rospy.spin()