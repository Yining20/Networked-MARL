import numpy as np
from tqdm import trange

# single agent, action number = 2, state number = 2
# The best policy in this toy setting is to choose action 1 in both states    
class ToyEnv:
    def __init__(self):
        self.trans_prob = np.array([[0.3,0.7],[0.2,0.8],[0.5,0.5],[0.3,0.7]]) # 4*2
        self.reward = np.array([0,1,1,2]) # 4*1

        self.GlobalState = 0
        self.newGlobalState = 0
        self.globalAction = 0
        self.globalReward = 0
    
    def initialize(self):
        self.GlobalState = np.random.choice(a = 2)

    def update_action(self, _action):
        self.globalAction = _action

    def generate_reward(self):
        self.newGlobalState = np.random.choice(a = 2, \
            p=self.trans_prob[self.GlobalState * 2 + self.globalAction,:])  

        self.globalReward = self.reward[self.GlobalState * 2 + self.globalAction]


    def step(self):
        self.GlobalState = self.newGlobalState

    def observe_state_g(self):
        result = self.GlobalState
        return result

    def observe_state_action_g(self):
        result = (self.GlobalState, self.globalAction)
        return result


    def observe_reward(self):
        return self.globalReward


'''
# test
if __name__ == "__main__":

    k = 1
    node_per_grid = 20
    gridNum = 1
    S = 20
    
    nodeNum = node_per_grid * gridNum
    T = 100

    env = ToyEnv()
    env.initialize()
    for i in range(nodeNum):
        _temp_action = np.random.choice([0,1])  #[0,1] action set
        env.update_action(i,_temp_action)
    
    for t in trange(T):
        env.generate_reward()
        env.step()  
        for i in range(nodeNum):
            _temp_action = np.random.choice([0,1]) 
            env.update_action(i,_temp_action)


        if t == 50:
            env.observe_state_g()
            env.observe_state_action_g()
            env.observe_reward()
'''