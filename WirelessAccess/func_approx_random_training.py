import math
from scipy import special
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import env_random
import os

np.random.seed(0)

class RandomNode:
    def __init__(self, index, k, Lq, La, node_per_grid, A, env):
        self.index = index
        self.Lq = Lq # feature vector length for action-value function approx.
        self.La = La # feature vector length for action
        
        self.state = []  # The list of local state at different time steps
        self.action = []  # The list of local actions at different time steps
        self.reward = []  # The list of local reward at different time steps
        self.avgreward = [0.0] # The list of average local reward at different time steps
        self.kHop = []  # The list to record the (state, action) pairs of k-hop neighbors
        self.qFeatureMtx = {}
        self.actionFeatureMtx = {}
        self.qparams = np.random.rand(self.Lq)
        self.params = np.random.rand(self.La)
        self.qparams_tilde = np.zeros(self.Lq)

        self.k = k
        self.node_per_grid = node_per_grid
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

    # initialize the local state (called at the beginning of the training process)
    def initialize_state(self):
        self.state.append(self.env.observe_state_g(self.index))  

    # At each time, user node observes the state of the whole grid
    def update_state(self):
        self.state.append(self.env.observe_state_g(self.index))

    # need to call this after the first time step
    def update_k_hop(self):
        self.kHop.append(self.env.observe_state_action_g(self.index))
    
    def update_action(self):
        # get the current state
        curr_State = self.state[-1]

        if curr_State not in self.actionFeatureMtx:
            self.actionFeatureMtx[curr_State] = np.random.uniform(size = (self.actionNum , self.La))
        # compute the probability vector
        probVec = special.softmax(np.matmul(self.actionFeatureMtx[curr_State], self.params))
        # randomly select an action based on probVec
        currentAction = self.actionList[np.random.choice(a=self.actionNum, p=probVec)]
        self.action.append(currentAction)
        self.env.update_action(self.index, currentAction)

    # oneHopNeighbors is a list of accessNodes
    def update_reward(self):
        currentReward = self.env.observe_reward(self.index)
        self.reward.append(currentReward)

    def update_avgreward(self,_alpha):
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
        

        self.qparams_tilde = self.qparams + _alpha * _grad * _temp
        

     

    # eta is the learning rate
    def update_params(self, _beta, avg_Q):
        # curr_State_localAction = (self.state[-2], self.action[-2])
        last_State = tuple(self.state[-2])            
        _temp = special.softmax(np.matmul(self.actionFeatureMtx[last_State], self.params))
        _temp2 = np.zeros(self.La)
        for i in range(self.actionNum):
            _temp2 += _temp[i] * self.actionFeatureMtx[last_State][i,:]
        _action_index = self.find_action_index(self.action[-2])
        _grad = self.actionFeatureMtx[last_State][_action_index,:] - _temp2 
        self.params += _beta * _grad * avg_Q

        
    # update consensus action-value function parameters
    def update_qparams_consensus(self, neighbor_nodes):
        _temp = np.zeros(self.Lq)
        for _neighbor in neighbor_nodes:
            _temp += self.env.CMtx_in[self.index, _neighbor.index] * _neighbor.qparams_tilde
        self.qparams = _temp
        

def update_qValue_out(neighbor_grids):
    avg_Q = 0.0
    for _grid in neighbor_grids:
        _temp = np.zeros(_grid[0].Lq)
        for _node in _grid:
            _temp += _node.qparams
            # _temp = _temp/len(_grid)
            avg_Q += np.matmul(_node.qFeatureMtx[_node.kHop[-1]],_temp)            
    return avg_Q

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


if __name__ == "__main__":
    k = 1
    node_per_grid = 6 # 20
    gridNum = 3
    S = 20 # 20
    A = 2 #[0,1] action set

    nodeNum = node_per_grid * gridNum
    
    
    env = env_random.RandomEnv(k = k, node_per_grid= node_per_grid, gridNum = gridNum, S = S)

    _scale = 1
    _alpha = lambda t: 1 / (t+1) ** 0.65 * _scale
    _beta = lambda t: 1 / (t+1) ** 0.85 * _scale     

    # action-value function approximation feature vector dimension
    Lq = 10
    La = 5
    Ep_Num = 1 #50
    T = 30000 #100
    evalInterval = 1     # evaluate the policy every evalInterval rounds (outer loop)

    accessNodeList = []
    for i in range(nodeNum):
        accessNodeList.append(RandomNode(index=i, k=k, Lq = Lq, La = La, \
            node_per_grid=node_per_grid, A = A, env=env))

    policyRewardList = []

    script_dir = os.path.dirname(__file__)
    with open(script_dir+'\\data/func_approx-Random-{}-{}-{}-{}-{}.txt'.format(gridNum, node_per_grid, S, A, k), 'w') as f:  # used to check the progress of learning
        # first erase the file
        f.seek(0)
        f.truncate()
     
    for e in trange(Ep_Num):
        env.initialize()
        for i in range(nodeNum):
            accessNodeList[i].restart()
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

        # test
        fig_temp= np.zeros([5,T])

        for t in trange(1,T+1): 
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

              
            avg_Q = []
            if t == 0:
                avg_Q = [0] * gridNum
            else:
                for j in range(gridNum):
                    neighbor_grids = []
                    for a in env.grid_net.find_neighbors(j, k):
                        _temp = []
                        for b in range(node_per_grid):
                            _temp.append(accessNodeList[a*node_per_grid+b])
                        neighbor_grids.append(_temp)
                    avg_Q.append(update_qValue_out(neighbor_grids)/nodeNum)
            

            for i in range(nodeNum):           
                accessNodeList[i].update_avgreward(_alpha(t))
                accessNodeList[i].update_q_params(_alpha(t))

            for i in range(nodeNum):
                accessNodeList[i].update_params(_beta(t), avg_Q[i//node_per_grid])
                # accessNodeList[i].update_params(_beta(t), \
                #     accessNodeList[i].calc_q(accessNodeList[i].kHop[-2]))


            for i in range(nodeNum):
                neighbor_nodes = []
                for j in env.node_net.find_neighbors(i % node_per_grid,1):
                    neighbor_nodes.append(accessNodeList[(i // node_per_grid) * node_per_grid + j])
                accessNodeList[i].update_qparams_consensus(neighbor_nodes)
            
            env.change_comm_net()

            

            # test
            fig_temp[:,t-1] = accessNodeList[0].qparams[0:5]

        # test
        fig, ax = plt.subplots()
        _range = np.arange(T)
        for i in range(5):
            ax.plot(_range, fig_temp[i,:], linewidth=1)
        plt.show()
        plt.title("Q parameters of agent 0")
        plt.xlabel('iteration')

        plt.plot(np.arange(T), accessNodeList[0].avgreward[0:T], linewidth=1)
        plt.title("Globally Averaged Return (mu)")
        plt.xlabel('iteration')
        plt.show()
            
        if e % evalInterval == evalInterval - 1:
            tempReward = eval_policy(node_list=accessNodeList, rounds=400, env=env)
            with open(script_dir+'\\data/func_approx-Random-{}-{}-{}-{}-{}.txt'.format(gridNum, node_per_grid, S, A, k), 'a') as f:  
                f.write("%f\n" % tempReward)
            policyRewardList.append(tempReward)

    

    lam = np.linspace(0, (len(policyRewardList) - 1) * evalInterval, len(policyRewardList))
    plt.plot(lam, policyRewardList)
    plt.savefig(script_dir+"\\data/func_approx-Random-{}-{}-{}-{}-{}.jpg".format(gridNum, node_per_grid, S, A, k))
    
   