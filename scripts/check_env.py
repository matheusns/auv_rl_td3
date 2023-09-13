#!/usr/bin/env python3.8

import rospy
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from auv_rl_gym.task_envs.desistek_saga.auto_docking import AutoDocking


class EnvChecker:
    def __init__(self):
        self.env = gym.make('DesistekSagaAutoDocking-v0')

    def check(self):
        check_env(self.env, warn=True)

if __name__ == "__main__":
    rospy.init_node('env_checker', anonymous=True)
    env_checker = EnvChecker()
    env_checker.check()
    rospy.loginfo("Environment checked and ready to use.")
    rospy.spin()