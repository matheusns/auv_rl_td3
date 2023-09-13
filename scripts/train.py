#!/usr/bin/env python3.8

import rospy
import gymnasium as gym
from stable_baselines3 import TD3
from auv_rl_gym.task_envs.desistek_saga.auto_docking import AutoDocking


class TD3Checker:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        self.env = gym.make('DesistekSagaAutoDocking-v0')
        self.model = TD3("MlpPolicy", self.env, verbose=1)

    def learn(self):
        self.model.learn(total_timesteps=self.timesteps)

if __name__ == "__main__":
    rospy.init_node('env_checker', anonymous=True)
    
    timesteps = rospy.get_param("/desistek_saga/learn/timesteps")

    td3 = TD3Checker(timesteps)
    initial_time = rospy.Time.now()
    td3.learn()
    elapsed_time = rospy.Time.now() - initial_time
    rospy.loginfo("Elapsed time: %s", elapsed_time)
    rospy.spin()
    
    
    import gymnasium as gym


env = gym.make("CartPole-v1")


vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()