import gym
import numpy as np
import random
import math
import matplotlib.pyplot as plt

NUM_EPISODES = 1000
MAX_T = 250
#업데이트비율
ALPHA = 0.7
#보상비율
GAMMA = 0.9
#랜덤한 확률로 움직이게한다
EXPLORATION_RATE = 0.99
EXPLORATION_RATE_DECAY = 0.75

env = gym.make('CartPole-v0')

NUM_ACTIONS = env.action_space.n


# S = {x좌표 ,세타, 속도, 각속도} 을 n개씩 나눈다.
CART_POS = np.linspace(-2.4, 2.4, 9)    #-2.4~2.4 를 4칸!
POLE_ANGLE = np.linspace(-1, 1, 9)      #-57도~57도
CART_VEL = np.linspace(-1, 1, 9)        
ANG_RATE = np.linspace(-1.5, -1.5, 9)

#S 의 상태는 모두 9^4개 
NUM_STATES = 9**4

DEBUG_MODE = True



#hit 개수
streaks = 0
#Q 상태모집 Q={x,theta,velocity,angular velocity,action}최대 2*9^4
Q = {}


#10개씩 잘랐으니, 어떤 수를 그 사이 값으로 찾아줌
def to_bins(value, bins):
        return np.digitize(x=[value], bins=bins)[0]

#관찰된state를 넣어주는 함수
def to_state(obs):
        x, theta, v, omega = obs
        state = (to_bins(x, CART_POS),to_bins(theta, POLE_ANGLE),to_bins(v, CART_VEL),to_bins(omega, ANG_RATE))
        return state

def get_action(state):
        p = np.random.uniform(0,1)
        #print p
        #가끔 딴짓을 해야함
  
        if p < EXPLORATION_RATE:
                return random.choice([0,1])
        x = []

        #왼족, 오른족중
        for action in [0,1]:
                if (state, action) in Q:
                        x.append(Q[(state, action)])
                else:
                        x.append(0)
             #   print(x)
        return np.argmax(x)


#저장한 데이터 읽어온다
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
                                print(state,action)
                                Q[(state,action)] = value
                        EXPLORATION_RATE = 0.25
        except IOError:
                EXPLORATION_RATE = 0.99
                print("No datafile")
#지금까지의 평균보다 큰놈만 잡아 알파와 감마를 갱신한다.
def higher(data_list):
        sum=0
        count=0
        for idx in range(0,NUM_EPISODES):
                sum += idx * data_list[idx]
                count += data_list[idx]
        if sum == 0:
                return 0
        return sum/count

        
if __name__ == "__main__":
        avg = 0
        avg_list=[]
        data_list = [0 for _ in range(NUM_EPISODES)]
        # 현재 들어간 성공횟수 리스트 


        input_from_txt()

        
        for episode in range(2000000):
                #초기화
                obs = env.reset()
                state = to_state(obs)
                ALPHA = 0.3
                GAMA = 0.5

                #1세트에 최대 몇번의 힘을 줄 것인가?
                for t in range(MAX_T):


                        """state_prime : 다음상태
                                state  : 현재상태
                        """
                        env.render()
                        action = get_action(state)

                        obs, reward, done, _ = env.step(action)

                        
                        state_prime = to_state(obs)
                        action_prime = get_action(state_prime)


                        ##Q(S,A) 조합이 처음인 경우
                        if (state_prime, action_prime) not in Q:
                                Q[(state_prime, action_prime)] = np.random.uniform(1,-1)
                                print("New Q value : ",state_prime,action_prime)
                        if (state, action) not in Q:
                                Q[(state, action)] = np.random.uniform(1,-1)
                                print("New Q value : ",state_prime,action_prime)

                        # done : 실패한 경우 episode => end
                        
                        if done :
                                print("Episode %d completed in %d" % (episode, t),"EXPLORATION_RATE : ", EXPLORATION_RATE)
                                avg += t
                                Q[(state, action)] = (1-ALPHA)*Q[(state, action)] + ALPHA*(-reward + GAMMA*Q[(state_prime, action_prime)])
                                data_list[t] += 1
                                if t > higher(data_list) :
                                        print(higher(data_list),t)
                                        ALPHA = 0.7
                                        GAMA = 1
                                if t > 199:
                                        streaks += 1
                                else:
                                        streaks = 0
                                break

                        
                                

                        # Q 러닝
                        Q[(state, action)] = (1-ALPHA)*Q[(state, action)] + ALPHA*(reward + GAMMA*Q[(state_prime, action_prime)])
                        #print ("newstate: ", Q[(state, action)])
                        state = state_prime


                        

                #랜덤하게 움직일 확률 갱신
                if(EXPLORATION_RATE < 0.01):
                        EXPLORATION_RATE = 0.25

                EXPLORATION_RATE *= EXPLORATION_RATE_DECAY
                
                if episode%100 == 0:
                        print ("Average of 100:", avg/100.0)
                        avg_list.append(avg/100)
                        print(avg_list)
                        avg = 0
                        #학습결를 result_Q 파일에 기록한다.
                        with open("result.txt","w") as f:
                                for word in Q:
                                    f.write(str(word))
                                    f.write("|"+str(Q[word])+"\n")
                                    print(word,Q[word])
                        #학습완료 
                if streaks >= 120:
                        print("Completed in %d episodes" % (episodes))
                        plt.plot(avg_list)
                        plt.show()
                        print(Q)
                        break
        



    
        plt.plot(avg_list)
        plt.show()


	


