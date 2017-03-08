import gym
import numpy as np
import random
import math
import matplotlib.pyplot as plt
#init
env = gym.make('CartPole-v0')



Q = {}

# S = {x좌표 ,세타, 속도, 각속도} 을 n개씩 나눈다.
CART_POS = np.linspace(-2.4, 2.4, 9)
POLE_ANGLE = np.linspace(-1, 1, 9)
CART_VEL = np.linspace(-1, 1, 9)
ANG_RATE = np.linspace(-1.5, 1.5, 9)
NUM_STATES = 9**4



#10개씩 잘랐으니, 어떤 수를 그 사이 값으로 찾아줌
def to_bins(value, bins):
        return np.digitize(x=[value], bins=bins)[0]

#관찰된state를 넣어주는 함수
def to_state(obs):
        x, theta, v, omega = obs
        state = (to_bins(x, CART_POS),to_bins(theta, POLE_ANGLE),to_bins(v, CART_VEL),to_bins(omega, ANG_RATE))
        return state
def get_action(state):
        x = []

        #왼족, 오른족중
        for action in [0,1]:
                if (state, action) in Q:
                        x.append(Q[(state, action)])
                else:
                        x.append(0)
             #   print(x)
        return np.argmax(x)
def input_from_txt():
        try:
                with open("result.txt",'r') as f:
                        for line in f.readlines():
                                line = line.split("|")
                                line = list(line)
                                state =[]
                                print(line[0])
                                for a in line[0]:
                                    if a in "0123456789":
                                        state.append(int(a))
                                value = float(line[1])
                                action = state.pop()
                                state = tuple(state)
                                Q[(state,action)] = value
                                print(state,action,value)
                        a = input()
        except IOError:
                print("No datafile")
if __name__ == "__main__":

                #데이터 불러오기
        input_from_txt()

        #최대 몇 세트?
        for episode in range(200):

                obs = env.reset()
                state = to_state(obs) 
            # 한세트에 몇번의 컨트롤?
                for t in range(200):
                        env.render()
                        
                        state = to_state(obs)
                        action = get_action(state)           
                        obs, reward, done,_ = env.step(action)
                        if (state, action) not in Q:
                                Q[(state, action)] = np.random.uniform(1,-1)                      
                        
                        #print(obs)
                            #[x좌표, 각도, 속도, 각속도]
                            #observation :[ x, angle, velocity, angular velocity]

                            #action : 0 왼, 1 오른
                            #action : 0 left, 1 right
                
                        if done:#math.fabs(obs[1]) > math.pi/4 or math.fabs(obs[0]) > 2.4:
                                print("Episode %d completed in %d" % (episode, t))
                                break

                        
                            #흐느적거리면 end
            
            


