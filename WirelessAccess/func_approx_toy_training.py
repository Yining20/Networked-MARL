import math
from scipy import special
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import env_toy
import os

# np.random.seed(0)

class ToyNode:
    def __init__(self, Lq, La, A, env):
        self.Lq = Lq # feature vector length for action-value function approx.
        self.La = La # feature vector length for action
        
        self.state = []  # The list of local state at different time steps
        self.action = []  # The list of local actions at different time steps
        self.reward = []  # The list of local reward at different time steps
        self.kHop = []
        self.avgreward = [0.0] # The list of average local reward at different time steps
        self.qFeatureMtx = {}
        self.actionFeatureMtx = {}
        self.qparams = np.random.rand(self.Lq)
        # self.params = np.random.rand(self.La)
        self.params = np.zeros(self.La)

    
        self.actionNum = A  # the number of possible actions
        self.actionList = [0, 1]  
        self.env = env

    def find_action_index(self, _action):
        _action_index = self.actionList.index(_action) 
        return _action_index

    def restart(self):
        self.state.clear()
        self.action.clear()
        self.reward.clear()
        self.avgreward.clear()
        self.kHop.clear()
        self.currentTimeStep = 0

    def restart_params(self):
        self.qparams = np.random.rand(self.Lq)
        self.params = np.zeros(self.La)

    # initialize the local state (called at the beginning of the training process)
    def initialize_state(self):
        self.state.append(self.env.observe_state_g())  

    # At each time, user node observes the state of the whole grid
    def update_state(self):
        self.state.append(self.env.observe_state_g())
    
    # need to call this after the first time step
    def update_k_hop(self):
        self.kHop.append(self.env.observe_state_action_g())

    def update_action(self):
        # get the current state
        curr_State = self.state[-1]

        if curr_State not in self.actionFeatureMtx:
            self.actionFeatureMtx[curr_State] = np.random.uniform(size = (self.actionNum , self.La))
        # compute the probability vector
        probVec = special.softmax(np.matmul(self.actionFeatureMtx[curr_State], self.params))
        # randomly select an action based on probVec
        currentAction = self.actionList[np.random.choice(a=self.actionNum, p=probVec)]
        # currentAction = self.actionList[np.random.choice(a=self.actionNum, p=[0,1])]
        self.action.append(currentAction)
        self.env.update_action(currentAction)

    # oneHopNeighbors is a list of accessNodes
    def update_reward(self):
        currentReward = self.env.observe_reward()
        self.reward.append(currentReward)

    def update_avgreward(self,_alpha):
        if len(self.avgreward) == 0:
            _temp = _alpha * self.reward[-1]
        else:
            _temp = (1-_alpha) * self.avgreward[-1] + _alpha * self.reward[-1]
        self.avgreward.append(_temp)

    def calc_q(self, StateAction_pair):
        Q_value = np.matmul(self.qFeatureMtx[StateAction_pair], self.qparams)
        return Q_value

    # kHopNeighbors is a list of accessNodes, alpha is learning rate
    def update_q_params(self, _alpha):
        last_StateAction = self.kHop[-2]
        curr_StateAction = self.kHop[-1]
        # fetch the Q value based on neighbors' states and actions
        lastQTerm1 = self.calc_q(last_StateAction)
        lastQTerm2 = self.calc_q(curr_StateAction)
        # compute the temporal difference
        _temp = self.reward[-1] - self.avgreward[-1] + lastQTerm2 - lastQTerm1
        # perform the Q value update
        _grad = self.qFeatureMtx[last_StateAction]
        

        self.qparams = self.qparams + _alpha * _grad * _temp
        


    # eta is the learning rate
    def update_params(self, _beta, avg_Q):
        # curr_State_localAction = (self.state[-2], self.action[-2])
        last_State = self.state[-2]          
        _temp = special.softmax(np.matmul(self.actionFeatureMtx[last_State], self.params))
        _temp2 = np.zeros(self.La)
        for i in range(self.actionNum):
            _temp2 += _temp[i] * self.actionFeatureMtx[last_State][i,:]
        _action_index = self.find_action_index(self.action[-2])
        _grad = self.actionFeatureMtx[last_State][_action_index,:] - _temp2 
        self.params += _beta * _grad * avg_Q

        

# do not update Q when evaluating a policy
def eval_policy(node_list, rounds, env):
    totalRewardSum = 0.0
    # for _ in range(rounds):
    env.initialize()
    for i in range(nodeNum):
        node_list[i].restart()
        node_list[i].initialize_state()
    for i in range(nodeNum):
        node_list[i].update_action()
    env.generate_reward()
    for i in range(nodeNum):
        node_list[i].update_reward()

    for _ in range(1, rounds + 1):
        env.step()
        for i in range(nodeNum):
            node_list[i].update_state()
        for i in range(nodeNum):
            node_list[i].update_action()
        env.generate_reward()
        for i in range(nodeNum):
            node_list[i].update_reward()
        # compute the total reward
        averageReward = 0.0
        for i in range(nodeNum):
            averageReward += node_list[i].reward[-1]
        averageReward /= nodeNum
        totalRewardSum += averageReward
    return totalRewardSum / rounds

def cal_avg_reward(env, policy):
    state_trans_prob = np.zeros((2,2)) # StateNum * StateNum
    for i in range(2):
        for j in range(2):
            state_trans_prob[i,:] += env.trans_prob[2*i+j,:]*policy[i,j]
    A = np.mat(state_trans_prob)
    A = np.eye(2)-A.T
    A[1,:] = [1,1]
    b = np.mat('0,1').T
    r = np.linalg.solve(A,b)
    r = np.reshape(np.array(r),2)
    expect_avg_reward = 0.0
    for i in range(2):
        for j in range(2):
            expect_avg_reward += r[i] * policy[i,j] * env.reward[2*i+j]
    return expect_avg_reward


if __name__ == "__main__":
    k = 1
    node_per_grid = 1 # 20
    gridNum = 1
    S = 2 
    A = 2 

    nodeNum = node_per_grid * gridNum
    
    
    env = env_toy.ToyEnv()

    _scale = 1
    # _alpha = lambda t: 1 / (t+1) ** 0.65 * _scale
    # _beta = lambda t: 1 / (t+1) ** 0.85 * _scale     
    _alpha = lambda t: 1 / (t+1) ** 0.5 * _scale
    _beta = lambda t: 1 / (t+1) ** 0.65 * _scale 

    # action-value function approximation feature vector dimension
    Lq = 5
    La = 5
    Ep_Num = 10 #50
    T = 30000 #100
    evalInterval = 1     # evaluate the policy every evalInterval rounds (outer loop)

    accessNodeList = []
    for i in range(nodeNum):
        accessNodeList.append(ToyNode(Lq = Lq, La = La, A = A, env=env))

    policyRewardList = []
    exp_avg_reward_list = []
    
    experi_avg_reward  = np.zeros(T+1)  
    # test
    _temp= np.zeros([5,T]) 
    for e in trange(Ep_Num):
        temp_experi_avg_reward = np.zeros((nodeNum,T+1))


        env.initialize()
        for i in range(nodeNum):
            accessNodeList[i].restart()
            accessNodeList[i].restart_params()
            accessNodeList[i].initialize_state()
        for i in range(nodeNum):
            accessNodeList[i].update_action()
        for i in range(nodeNum):
            accessNodeList[i].update_k_hop()

        # Synchronize Q feature Matrix after initialization
        for j in range(gridNum):
            _node = accessNodeList[j * node_per_grid]
            StateAction_pair = _node.kHop[-1]
            if StateAction_pair not in _node.qFeatureMtx:
                _node.qFeatureMtx[StateAction_pair] = np.random.uniform(size = _node.Lq)
                for l in range(1, node_per_grid):
                    accessNodeList[j * node_per_grid + l].qFeatureMtx[StateAction_pair] = _node.qFeatureMtx[StateAction_pair]

        

        for t in range(1,T+1): 
            env.generate_reward()
            for i in range(nodeNum):
                accessNodeList[i].update_reward()
            env.step()

            for i in range(nodeNum):
                accessNodeList[i].update_state()
            for i in range(nodeNum):
                accessNodeList[i].update_action()
            for i in range(nodeNum):
                accessNodeList[i].update_k_hop()
            # Synchronize Q feature Matrix
            for j in range(gridNum):
                _node = accessNodeList[j * node_per_grid]
                StateAction_pair = _node.kHop[-1]
                if StateAction_pair not in _node.qFeatureMtx:
                    _node.qFeatureMtx[StateAction_pair] = np.random.uniform(size = _node.Lq)
                    for l in range(1, node_per_grid):
                        accessNodeList[j * node_per_grid + l].qFeatureMtx[StateAction_pair] = _node.qFeatureMtx[StateAction_pair]


            for i in range(nodeNum):           
                accessNodeList[i].update_avgreward(_alpha(t))
                accessNodeList[i].update_q_params(_alpha(t))

            for i in range(nodeNum):
                accessNodeList[i].update_params(_beta(t), \
                    accessNodeList[i].calc_q(accessNodeList[i].kHop[-2]))
            
            # test
            t_start = 0
            if e == 0:
                policy = np.zeros((2,2))
                if len(accessNodeList[0].actionFeatureMtx) == 2:
                    if t < t_start:
                        t_start = t
                    policy[0,:] = special.softmax(np.matmul(accessNodeList[0].actionFeatureMtx[0], accessNodeList[0].params))
                    policy[1,:] = special.softmax(np.matmul(accessNodeList[0].actionFeatureMtx[1], accessNodeList[0].params))
                    exp_avg_reward_list.append(cal_avg_reward(env, policy)) 
              
                
                _temp[:,t-1] = accessNodeList[0].qparams[0:5]

            for i in range(nodeNum):
                temp_experi_avg_reward[i,t] = ((t-1)*temp_experi_avg_reward[i,t-1]+accessNodeList[i].reward[-1])/t
        
        temp_experi_avg_reward = np.sum(temp_experi_avg_reward, axis = 0)/ nodeNum
        experi_avg_reward = experi_avg_reward + temp_experi_avg_reward
    experi_avg_reward = experi_avg_reward / Ep_Num
        

        
    aa = special.softmax(np.matmul(accessNodeList[0].actionFeatureMtx[1], accessNodeList[0].params))
    print(aa)
    bb = special.softmax(np.matmul(accessNodeList[0].actionFeatureMtx[0], accessNodeList[0].params))
    print(bb)

    # test
    plt.figure()
    plt.plot(np.arange(T), accessNodeList[0].avgreward[0:T], linewidth=1)
    plt.title("Globally Average Return (mu)")
    plt.xlabel('iteration')

    
    plt.figure()
    plt.plot(np.arange(t_start,T), exp_avg_reward_list, linewidth=1)
    plt.title("Expected Average Return")
    plt.xlabel('iteration')
    # plt.show()

    plt.figure()
    plt.plot(np.arange(T), experi_avg_reward[1:], linewidth=1)
    plt.title("Experiemental Average Return")
    plt.xlabel('iteration')
    # plt.show()


    fig, ax = plt.subplots()
    _range = np.arange(T)
    for i in range(5):
        ax.plot(_range, _temp[i,:], linewidth=1)
    plt.show()
    plt.title("Q parameters of agent 0")
    plt.xlabel('iteration')

        
        
        

    
    
   