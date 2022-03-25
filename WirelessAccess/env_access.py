import numpy as np


class GlobalNetwork:
    def __init__(self, node_num, k):
        self.nodeNum = node_num  # the total number of nodes in this network
        # initialize the adjacency matrix of the global network
        self.adjacencyMatrix = np.eye(self.nodeNum, dtype=int)
        self.k = k  # the number of hops used in learning
        # cache the powers of the adjacency matrix
        self.adjacencyMatrixPower = [np.eye(self.nodeNum, dtype=int)]
        # use a hashmap to store the ((node, dist), neighbors) pairs which we have computed
        self.neighborDict = {}
        self.addingEdgesFinished = False  # have we finished adding edges?

    # add an undirected edge between node i and j
    def add_edge(self, i, j):
        self.adjacencyMatrix[i, j] = 1
        self.adjacencyMatrix[j, i] = 1

    # finish adding edges, so we can construct the k-hop neighborhood after adding edges
    def finish_adding_edges(self):
        _temp = np.eye(self.nodeNum, dtype=int)
        # the d-hop adjacency matrix is stored in self.adjacencyMatrixPower[d]
        for _ in range(self.k):
            _temp = np.matmul(_temp, self.adjacencyMatrix)
            self.adjacencyMatrixPower.append(_temp)
        self.addingEdgesFinished = True

    # query the d-hop neighborhood of node i, return a list of node indices.
    def find_neighbors(self, i, d):
        if not self.addingEdgesFinished:
            print("Please finish adding edges before call findNeighbors!")
            return -1
        if (i, d) in self.neighborDict:  # if we have computed the answer before, return it
            return self.neighborDict[(i, d)]
        neighbors = []
        for j in range(self.nodeNum):
            # this element > 0 implies that dist(i, j) <= d
            if self.adjacencyMatrixPower[d][i, j] > 0:
                neighbors.append(j)
        # cache the answer so we can reuse later
        self.neighborDict[(i, d)] = neighbors
        return neighbors




class AccessNetwork(GlobalNetwork):
    def __init__(self, node_num, k, access_num):
        super(AccessNetwork, self).__init__(node_num, k)
        self.accessNum = access_num
        self.accessMatrix = np.zeros((node_num, access_num), dtype=int)
        self.transmitProb = np.ones(access_num)  
        self.serviceNum = np.zeros(access_num, dtype=int)  # how many agents should I provide service to?

    # add an access point a for node i
    def add_access(self, i, a):
        self.accessMatrix[i, a] = 1
        self.serviceNum[a] += 1

    # finish adding access points. we can construct the neighbor graph
    def finish_adding_access(self):
        # use accessMatrix to construct the adjacency matrix of (user) nodes
        self.adjacencyMatrix = np.matmul(self.accessMatrix, np.transpose(self.accessMatrix))
        super(AccessNetwork, self).finish_adding_edges() 

    # find the access points for node i
    def find_access(self, i):
        access_points = []
        for j in range(self.accessNum):
            if self.accessMatrix[i, j] > 0:
                access_points.append(j)
        return access_points

    # set transmission probability
    def set_transmit_prob(self, transmit_prob):
        self.transmitProb = transmit_prob

def construct_grid_network(node_num, width, height, k, node_per_grid, transmit_prob):
    if node_num != width * height * node_per_grid:
        print("nodeNum does not satisfy the requirement of grid network!")
        return None
    access_num = (width - 1) * (height - 1)
    grid_num = width * height
    grid_net = AccessNetwork(node_num=grid_num, access_num=access_num, k=k)
    node_net = GlobalNetwork(node_num=node_num, k = 1)
    for j in range(access_num):
        upper_left = j // (width - 1) * width + j % (width - 1)
        upper_right = upper_left + 1
        lower_left = upper_left + width
        lower_right = lower_left + 1
        accessed_nodes = [upper_left, upper_right, lower_left, lower_right]
        for a in accessed_nodes:
            grid_net.add_access(a, j)
    grid_net.finish_adding_access()       
    for i in range(grid_num):
        for b in range(node_per_grid):
            for c in range(b, node_per_grid):
                node_net.add_edge(node_per_grid * i + b, node_per_grid * i + c)
    node_net.finish_adding_edges()
    # setting transmitProb
    if transmit_prob == 'allone':
        transmit_prob = np.ones(access_num)
    elif transmit_prob == 'random':
        np.random.seed(0)
        transmit_prob = np.random.rand(access_num)
    grid_net.set_transmit_prob(transmit_prob)
    return grid_net, node_net


class AccessGridEnv:
    def __init__(self, height, width, k, node_per_grid=1, transmit_prob='allone', ddl=1, arrival_prob=0.5):
        self.height = height
        self.width = width
        self.k = k
        self.nodePerGrid = node_per_grid
        self.transmitProb = transmit_prob
        self.ddl = ddl
        self.arrivalProb = arrival_prob

        self.gridNum = height * width
        self.nodeNum = height * width * node_per_grid
        self.accessNum = (height - 1) * (width - 1)
        # the i th row is the state of agent i
        self.GlobalState = np.zeros((self.nodeNum, self.ddl), dtype=int)
        self.newGlobalState = np.zeros((self.nodeNum, self.ddl), dtype=int)
        self.grid_GlobalState = np.zeros((self.gridNum, self.ddl), dtype=int)
        # self.grid_newGlobalState = np.zeros((self.gridNum, self.ddl), dtype=int)
        self.globalAction = np.zeros(
            (self.nodeNum, 2), dtype=int)  # (slot, accessPoint)
        self.globalReward = np.zeros(self.nodeNum, dtype=float)

        self.grid_net, self.node_net = construct_grid_network(self.nodeNum, self.width, self.height, self.k, self.nodePerGrid,
                                                    self.transmitProb)                            
        self.CMtx_in = np.zeros([self.nodeNum,self.nodeNum])
        self.Build_CMtx_in()

    # Call at start of each episode. Packets with deadline ddl arrive in the buffers according to arrivalProb
    def initialize(self):
        lastCol = np.random.binomial(n=1, p=self.arrivalProb, size=self.nodeNum)
        self.GlobalState = np.zeros((self.nodeNum, self.ddl), dtype=int)
        self.GlobalState[:, self.ddl - 1] = lastCol         
        self.grid_GlobalState = self.generate_grid_state(self.GlobalState)
        self.globalReward = np.zeros(self.nodeNum, dtype=float)

    def generate_grid_state(self, State_Mtx):
        Grid_State_Mtx = np.zeros((self.gridNum, self.ddl), dtype=int)
        for i in range(self.gridNum):
            Grid_State_Mtx[i,:] = np.sum(State_Mtx[i* self.nodePerGrid:(i+1) * self.nodePerGrid,:],axis=0)
        return Grid_State_Mtx
    
    def observe_state_g(self, index, depth='None'):
        if depth == 'None':
            depth = self.k
        result = []
        for j in self.grid_net.find_neighbors(index//self.nodePerGrid, depth):
            result.append(tuple(self.grid_GlobalState[j, :]))
        return tuple(result)

    def observe_state_g_v2(self, index, depth='None'):
        if depth == 'None':
            depth = self.k
        result = []
        for j in self.grid_net.find_neighbors(index//self.nodePerGrid, depth):
            result.append(self.grid_GlobalState[j, 0])
        return tuple(result)

    # def observe_local_state_g(self, index):
    #     result = self.GlobalState[index, :]
    #     return tuple(result)

    def observe_local_state_g(self, index, depth='None'):
        if depth == 'None':
            depth = self.k
        result = []
        for a in self.grid_net.find_neighbors(index//self.nodePerGrid, depth):
            for b in range(self.nodePerGrid):
                result.append(tuple(self.GlobalState[a * self.nodePerGrid + b, :]))
        return tuple(result)

    def observe_local_state_g_v2(self, index, depth='None'):
        if depth == 'None':
            depth = self.k
        result = []
        for a in self.grid_net.find_neighbors(index//self.nodePerGrid, depth):
            for b in range(self.nodePerGrid):
                result.append(self.GlobalState[a * self.nodePerGrid + b, 0])
        return tuple(result)

    def observe_state_action_g(self, index, depth = 'None'):
        if depth == 'None':
            depth = self.k
        result = []
        for j in self.grid_net.find_neighbors(index//self.nodePerGrid, depth):
            grid_action = []
            for i in range(self.nodePerGrid):
                grid_action.append((self.globalAction[j*self.nodePerGrid + i, 0],\
                    self.globalAction[j*self.nodePerGrid + i, 1])) 
            result.append((tuple(self.grid_GlobalState[j, :]), tuple(grid_action)))
        return tuple(result)

    def observe_local_state_action_g(self, index, depth = 'None'):
        if depth == 'None':
            depth = self.k
        result = []
        for j in self.grid_net.find_neighbors(index//self.nodePerGrid, depth):
            for i in range(self.nodePerGrid):
                result.append((tuple(self.GlobalState[j * self.nodePerGrid + i, :]), \
                    self.globalAction[j*self.nodePerGrid + i, 0],\
                        self.globalAction[j*self.nodePerGrid + i, 1])) 
        return tuple(result)

    def observe_reward(self, index):
        return self.globalReward[index]

    def generate_reward(self):
        # reset the global reward
        self.globalReward = np.zeros(self.nodeNum, dtype=float)
        self.newGlobalState = self.GlobalState
        clientCounter = - np.ones(self.accessNum, dtype=int)
        # bind client to access points
        for i in range(self.nodeNum):
            slot = self.globalAction[i, 0]
            accessPoint = self.globalAction[i, 1]
            """
            if slot >= 0 and self.GlobalState[i, slot] == 0: #the client is not sending out anything
                continue
            """
            if accessPoint == -1:  # the client does not send out a message
                continue
            if clientCounter[accessPoint] == -1:  # if nobody binds to the access point, bind the client to it
                clientCounter[accessPoint] = i
            elif clientCounter[accessPoint] >= 0:  # somebody has already bind to the access point, crash
                clientCounter[accessPoint] = -2
        # assign rewards & set GlobalState
        for a in range(self.accessNum):
            if clientCounter[a] >= 0:  # a client successfully bind to the access point
                client = clientCounter[a]
                # check if the message is valid
                # print("client: ", client)
                slot = self.globalAction[client, 0]
                if self.GlobalState[client, slot] == 1:  # this is a valid message
                    success = np.random.binomial(1, self.grid_net.transmitProb[a])
                    if success == 1:
                        self.newGlobalState[client, slot] -= 1
                        self.globalReward[client] = 1.0
        # update global state to next time step
        lastCol = np.random.binomial(n=1, p=self.arrivalProb, size=self.nodeNum)
        self.newGlobalState[:, 0:(self.ddl - 1)] = self.newGlobalState[:, 1:self.ddl]
        self.newGlobalState[:, self.ddl - 1] = lastCol

    def step(self):
        self.GlobalState = self.newGlobalState
        self.grid_GlobalState = self.generate_grid_state(self.GlobalState)

    def update_action(self, index, action):
        slot, access_point = action
        self.globalAction[index, 0] = slot
        self.globalAction[index, 1] = access_point

    def Build_CMtx_in(self):
        _degree=[]
        for i in range(self.nodeNum):
            _degree.append(len(self.node_net.find_neighbors(i,1))-1)
        for i in range(self.nodeNum):
            for j in self.node_net.find_neighbors(i, 1):
                if i != j:
                    self.CMtx_in[i,j] = 1 / (max(_degree[i],_degree[j]) + 1)
            self.CMtx_in[i,i] = 1 - sum(self.CMtx_in[i])
