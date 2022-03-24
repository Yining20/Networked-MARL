import math
from scipy import special
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import env_access
import os

np.random.seed(0)

class AccessNode:
    def __init__(self, index, deadline, k, Lq, La, node_per_grid, env):
        self.index = index
        self.Lq = Lq # feature vector length for action-value function approx.
        self.La = La # feature vector length for action
        
        self.state = []  # The list of local state at different time steps
        self.action = []  # The list of local actions at different time steps
        self.reward = []  # The list of local reward at different time steps
        self.avgreward = [] # The list of average local reward at different time steps
        self.kHop = []  # The list to record the (state, action) pairs of k-hop neighbors
        self.qFeatureMtx = {}
        self.actionFeatureMtx = {}
        self.qparams = np.random.rand(self.Lq)
        self.params = np.zeros(self.La) #np.random.rand(self.La)
        self.qparams_tilde = np.zeros(self.Lq)

        self.k = k
        self.ddl = deadline  # the initial deadline of each packet
        self.node_per_grid = node_per_grid
        self.accessPoints = env.grid_net.find_access(
            i= index//node_per_grid)  # find and cache the access points this node can access
        self.accessNum = len(self.accessPoints)  # the number of access points
        self.actionNum = self.accessNum * self.ddl + 1  # the number of possible actions, action is a tuple (slot, accessPoint)
        # construct a list of possible actions
        self.actionList = [(-1, -1)]  # (-1, -1) is an empty action that does nothing
        for slot in range(self.ddl):
            for a in self.accessPoints:
                self.actionList.append((slot, a))
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

    def restart_params(self):
        self.qparams = np.random.rand(self.Lq)
        self.params = np.zeros(self.La)

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
            self.actionFeatureMtx[curr_State] = np.random.uniform(size = (self.actionNum, self.La))
        # compute the probability vector
        probVec = special.softmax(np.matmul(self.actionFeatureMtx[curr_State], self.params))
        # randomly select an action based on probVec
        currentAction = self.actionList[np.random.choice(a=self.actionNum, p=probVec)]
        '''
        if self.index == 0:
            currentAction = self.actionList[np.random.choice(a=self.actionNum, p=[0,1])]
        else:
            currentAction = self.actionList[np.random.choice(a=self.actionNum, p=[1,0])]
        '''
        self.action.append(currentAction)
        self.env.update_action(self.index, currentAction)

    # oneHopNeighbors is a list of accessNodes
    def update_reward(self):
        currentReward = self.env.observe_reward(self.index)
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
        _temp= np.zeros(self.Lq)
        for _neighbor in neighbor_nodes:
            _temp += self.env.CMtx_in[self.index, _neighbor.index] * _neighbor.qparams_tilde
        self.qparams = _temp


    def update_action_benchmark(self, benchmark_policy):
        actProb = benchmark_policy[0]
        flagAct = np.random.binomial(1, actProb)  # should I send out a packet?
        if flagAct == 0:
            self.action.append((-1, -1))
            self.env.update_action(self.index, (-1, -1))
            return
        # find the packet with the earliest ddl
        benchSlot = -1
        local_state_temp = env.observe_local_state_g(self.index)
        for i in range(self.ddl):
            if local_state_temp[i] > 0:
                benchSlot = i
                break
        if benchSlot == -1:
            self.action.append((-1, -1))
            self.env.update_action(self.index, (-1, -1))
            return
        # select the access point to send to
        benchProb = benchmark_policy[1:]
        benchAccessPoint = self.accessPoints[np.random.choice(a=self.accessNum, p=benchProb)]
        self.action.append((benchSlot, benchAccessPoint))
        self.env.update_action(self.index, (benchSlot, benchAccessPoint))
        return        

def update_qValue_out(neighbor_grids):
    avg_Q = 0.0
    for _grid in neighbor_grids:
        for _node in _grid:
            _temp = _node.qparams
            # _temp = _temp/len(_grid)
            avg_Q += np.matmul(_node.qFeatureMtx[_node.kHop[-1]],_temp)

    return avg_Q

def eval_benchmark(node_list, rounds, T, act_prob, env):
    totalRewardSum = np.zeros(T)
    benchmarkPolicyList = []
    _scale = 1
    _alpha = lambda t: 1 / (t+1) ** 0.65 * _scale
    for i in range(gridNum):
        accessPoints = env.grid_net.find_access(i)
        accessPointsNum = len(accessPoints)
        benchmarkPolicy = np.zeros(accessPointsNum + 1)
        totalSum = 0.0
        for j in range(accessPointsNum):
            tmp = 100 * env.grid_net.transmitProb[accessPoints[j]] \
                / env.grid_net.serviceNum[accessPoints[j]] \
                    / node_per_grid
            totalSum += tmp
            benchmarkPolicy[j + 1] = tmp 
        benchmarkPolicy[1:] = benchmarkPolicy[1:]/totalSum 
        benchmarkPolicy[0] = act_prob
        benchmarkPolicyList.append(benchmarkPolicy)

    for _ in range(1, rounds+1):
        env.initialize()
        temp_RewardSum = np.zeros((nodeNum,T))
        for i in range(nodeNum):
            node_list[i].restart()
            node_list[i].initialize_state()

        for i in range(nodeNum):
            node_list[i].update_action_benchmark(benchmarkPolicyList[i//node_per_grid])
        

        for t in range(1, T + 1):
            env.generate_reward()
            for i in range(nodeNum):
                node_list[i].update_reward()
                node_list[i].update_avgreward(_alpha(t))
            env.step()
            for i in range(nodeNum):
                node_list[i].update_state()
            for i in range(nodeNum):
                node_list[i].update_action_benchmark(benchmarkPolicyList[i//node_per_grid])
        # compute the total reward
        
            for i in range(nodeNum):
                if t == 1:
                    temp_RewardSum[i,t-1] = node_list[i].reward[-1]
                else:
                    temp_RewardSum[i,t-1] = ((t-1)*temp_RewardSum[i,t-2]+node_list[i].reward[-1])/t
        temp_RewardSum = np.sum(temp_RewardSum, axis = 0)/ nodeNum
        totalRewardSum = totalRewardSum + temp_RewardSum
       
    
    return totalRewardSum / rounds

# do not update Q when evaluating a policy
def eval_policy(node_list, rounds, env):
    totalRewardSum = 0.0
    
    # for _ in range(rounds):
    env.initialize()
    for i in range(nodeNum):
        # node_list[i].restart()
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
    # height = 3
    # width = 4
    # node_per_grid = 2

    ddl = 1

    height = 2
    width = 2
    node_per_grid = 1
    gridNum = height * width
    nodeNum = gridNum * node_per_grid
    env = env_access.AccessGridEnv(height=height, width=width, k=k, node_per_grid=node_per_grid, ddl = ddl)
    

    _scale = 1
    _alpha = lambda t: 1 / (t+1) ** 0.65 * _scale
    _beta = lambda t: 1 / (t+1) ** 0.85 * _scale     

    # action-value function approximation feature vector dimension
    Lq = 50
    La = 30
    Ep_Num = 10
    T = 30000 #20000
    evalInterval = 1     # evaluate the policy every evalInterval rounds (outer loop)

    accessNodeList = []
    for i in range(nodeNum):
        accessNodeList.append(AccessNode(index=i, deadline = ddl, k=k, Lq = Lq, La = La, node_per_grid=node_per_grid, env=env))

    policyRewardList = []

    script_dir = os.path.dirname(__file__)
    with open(script_dir+'\\data/Tabular-Access-{}-{}-{}.txt'.format(height, width, k), 'w') as f:  # used to check the progress of learning
        # first erase the file
        f.seek(0)
        f.truncate()

    env.initialize()
    experi_avg_reward  = np.zeros(T)  
    # Actor-critic
    for e in trange(Ep_Num):
        temp_experi_avg_reward = np.zeros((nodeNum,T))
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

        # test
        fig_temp= np.zeros([5,T])

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
                accessNodeList[i].update_params(_beta(t),avg_Q[i//node_per_grid])

            for i in range(nodeNum):
                neighbor_nodes = []
                for j in env.node_net.find_neighbors(i,1):
                    neighbor_nodes.append(accessNodeList[j])
                accessNodeList[i].update_qparams_consensus(neighbor_nodes)

            # test
            fig_temp[:,t-1] = accessNodeList[0].qparams[0:5]

            for i in range(nodeNum):
                if t == 1:
                    temp_experi_avg_reward[i,t-1] = accessNodeList[i].reward[-1]
                else:
                    temp_experi_avg_reward[i,t-1] = ((t-1)*temp_experi_avg_reward[i,t-2]+accessNodeList[i].reward[-1])/t

        
        # Benchmark Aloha policy
        if e == 0:
            bestBenchmark = 0.0
            bestBenchmarkProb = 0.0
            bestBenchmarkTrace = np.zeros(T)
            for i in range(10):
                tmp = eval_benchmark(node_list=accessNodeList, rounds=1, T=T, act_prob=i / 10.0, env=env)
                if tmp[-1] > bestBenchmark:
                    bestBenchmark = tmp[-1]
                    bestBenchmarkTrace = tmp
                    bestBenchmarkProb = i / 10.0
        
        
        temp_experi_avg_reward = np.sum(temp_experi_avg_reward, axis = 0)/ nodeNum
        experi_avg_reward = experi_avg_reward + temp_experi_avg_reward
    experi_avg_reward = experi_avg_reward / Ep_Num

    # test
    fig, ax = plt.subplots()
    _range = np.arange(T)
    for i in range(5):
        ax.plot(_range, fig_temp[i,:], linewidth=1)
    plt.title("Q parameters of agent 0")
    plt.xlabel('iteration')
    plt.show()

    
    fig2, ax2 = plt.subplots()
    ax2.plot(np.arange(T), experi_avg_reward, linewidth=1, label='Scalable Actor-critic')
    ax2.plot(np.arange(T), bestBenchmark*np.ones(T), label='Benchmark ALOHA')
    ax2.legend()
    plt.title("Experiemental Averaged Return")
    plt.xlabel('iteration')
    plt.show()


    lam = np.linspace(0, (len(policyRewardList) - 1) * evalInterval, len(policyRewardList))
    plt.plot(lam, policyRewardList)
    plt.savefig(script_dir+"\\data/Tabular-Access-{}-{}-{}.jpg".format(height, width, k))
   
    fig2, ax2 = plt.subplots()
    ax2.plot(np.arange(T), experi_avg_reward, linewidth=1, label='Scalable Actor-critic')
    ax2.plot(np.arange(T), bestBenchmark*np.ones(T), label='Benchmark ALOHA')
    ax2.plot(np.arange(T), temp_experi_avg_reward, label='1Ep, Scalable Actor-critic')
    ax2.legend()
    plt.title("Experiemental Averaged Return (\mu)")
    plt.xlabel('iteration')
    plt.show()