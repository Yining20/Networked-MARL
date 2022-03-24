import numpy as np
from tqdm import trange

np.random.seed(0)

class Random_Net:
    def __init__(self, nodeNum, edgeNum):
        self.nodeNum = nodeNum
        self.edgeNum = edgeNum
        self.adjacencyMatrix = np.eye(self.nodeNum, dtype=int)
        self.contruct_random_net()
        self.adjacencyMatrixPower = [np.eye(self.nodeNum, dtype=int)]
        
        
    def find_neighbors(self, index, k):
        if len(self.adjacencyMatrixPower) < k+1:
            for _ in range(len(self.adjacencyMatrixPower),k+1):
                _temp = np.matmul(self.adjacencyMatrixPower[-1], self.adjacencyMatrix)
                self.adjacencyMatrixPower.append(_temp)

        neighbors = []
        for i in range(self.nodeNum):
            if self.adjacencyMatrixPower[k][index, i] == 1:
                neighbors.append(i)
        return neighbors

    def contruct_random_net(self):
        all_edges = []
        for i in range(self.nodeNum):
            for j in range(i+1, self.nodeNum):
                all_edges.append((i, j))
        np.random.shuffle(all_edges)
        all_edges = all_edges[:self.edgeNum]
        for _edge in all_edges:
            self.adjacencyMatrix[_edge[0],_edge[1]] = 1
            self.adjacencyMatrix[_edge[1],_edge[0]] = 1
        for i in range(self.nodeNum):
            if sum(self.adjacencyMatrix[i,:]) == 0:
                _temp = np.random.choice([i for i in range(self.nodeNum)])
                if _temp == i:
                    _temp += 1
                self.adjacencyMatrix[i, _temp] = 1
                self.adjacencyMatrix[_temp, i] = 1
        

class RandomEnv:
    def __init__(self, k, node_per_grid, gridNum, S):
        self.node_per_grid = node_per_grid
        self.gridNum = gridNum
        self.nodeNum = gridNum * node_per_grid
        self.k = k # k-hop of regions
        self.S = S # state number for each region
        self.trans_prob = {}
        self.reward_dict = {}
        self.grid_net = Random_Net(gridNum,2*(gridNum - 1))
        self.node_net = Random_Net(node_per_grid,2*(node_per_grid - 1))
        self.CMtx_in = np.zeros([self.nodeNum,self.nodeNum])
        self.Build_CMtx_in()
        
        self.GlobalState = np.zeros(self.gridNum, dtype = int)
        self.newGlobalState = np.zeros(self.gridNum, dtype = int)
        self.globalAction = np.zeros(self.nodeNum,dtype = int)
        self.globalReward = np.zeros(self.nodeNum, dtype=float)
    
    def initialize(self):
        self.GlobalState = np.random.choice(a = self.S, size = self.gridNum)

    def change_comm_net(self):
        self.node_net.adjacencyMatrix = np.eye(self.nodeNum, dtype=int)
        self.node_net.contruct_random_net()
        self.node_net.adjacencyMatrixPower = [np.eye(self.nodeNum, dtype=int)]
        self.CMtx_in = np.zeros([self.nodeNum,self.nodeNum])
        self.Build_CMtx_in()

    def update_action(self, _index, _action):
        self.globalAction[_index] = _action

    def generate_reward(self):
        for i in range(self.gridNum):
            local_state = []
            local_action = []
            for j in self.grid_net.find_neighbors(i, self.k):
                local_state.append(self.GlobalState[j])
                for _node_j in range(self.node_per_grid):
                    local_action.append(self.globalAction[j*self.node_per_grid+_node_j])
            
            # Update new state
            if (tuple(local_state), tuple(local_action)) not in self.trans_prob:
                _temp = np.random.uniform(size = (self.S, self.gridNum)) + 1e-5
                _temp = _temp / np.sum(_temp,axis = 0)
                self.trans_prob[(tuple(local_state), tuple(local_action))] = _temp 

            self.newGlobalState[i] = np.random.choice([i for i in range(self.S)], \
                p=self.trans_prob[(tuple(local_state), tuple(local_action))][:,i])  

            # Generate reward
            if (tuple(local_state), tuple(local_action)) not in self.reward_dict:
                self.reward_dict[(tuple(local_state), tuple(local_action))] = \
                    np.random.uniform(low=0.0, high=4.0, size=self.node_per_grid)
           
            self.globalReward[i * self.node_per_grid : (i+1) * self.node_per_grid] = \
                self.reward_dict[(tuple(local_state), tuple(local_action))] + \
                    np.random.uniform(low=-0.5, high=0.5, size=self.node_per_grid)


    def step(self):
        self.GlobalState = self.newGlobalState

    def observe_state_g(self, index):
        result = []
        grid_index = index // self.node_per_grid   
        for j in self.grid_net.find_neighbors(grid_index, self.k):
            result.append(self.GlobalState[j])
        return tuple(result)

    def observe_state_action_g(self,index):
        local_state = []
        local_action = []
        grid_index = index // self.node_per_grid
        for j in self.grid_net.find_neighbors(grid_index, self.k):
            local_state.append(self.GlobalState[j])
            for _node_j in range(self.node_per_grid):
                local_action.append(self.globalAction[j*self.node_per_grid+_node_j])
        return (tuple(local_state), tuple(local_action))


    def observe_reward(self, index):
        return self.globalReward[index]


    def Build_CMtx_in(self):
        _degree = []
        _temp = np.zeros([self.node_per_grid,self.node_per_grid])
        for i in range(self.node_per_grid):
            _degree.append(len(self.node_net.find_neighbors(i,1))-1)
        for i in range(self.node_per_grid):
            for j in self.node_net.find_neighbors(i, 1):
                if i != j:
                    _temp[i,j] = 1 / (max(_degree[i],_degree[j]) + 1)
            _temp[i,i] = 1 - sum(_temp[i,:])
        for i in range(self.gridNum):
            self.CMtx_in[i * self.node_per_grid : (i+1) * self.node_per_grid , \
                i * self.node_per_grid : (i+1) * self.node_per_grid] =_temp


# test
if __name__ == "__main__":
    '''
    k = 1
    node_per_grid = 10
    gridNum = 3
    S = 3
    '''

    k = 1
    node_per_grid = 20
    gridNum = 1
    S = 20
    
    nodeNum = node_per_grid * gridNum
    T = 100

    env = RandomEnv(k = k, node_per_grid= node_per_grid, gridNum = gridNum, S = S)
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
            env.grid_net.adjacencyMatrix
            env.node_net.adjacencyMatrix
            index = 9
            env.Build_CMtx_in()
            env.observe_state_g(index)
            env.observe_state_action_g(index)
            env.observe_reward(index)
