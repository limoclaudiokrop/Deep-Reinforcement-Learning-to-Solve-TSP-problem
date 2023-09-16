import time

def readData():
    states = []
    long_lats = []
    prizes = []
    with open("file.txt") as file:
        for line in file:
            words = line.split(',')
            longs = words[1].split()
            
            arr = []
            states.append(words[0] + "," + longs[0])
            arr.append(float(longs[1]))
            arr.append(float(longs[2]))
            prizes.append(float(longs[3]))
            long_lats.append(arr)
            
    return states, long_lats, prizes

states, long_lats, prizes = readData()


def calcDistance(c1, c2):
    return ((c1[0] - c2[0])**2 + (c1[1]-c2[1])**2)**0.5

def getDistanceBtwCities(c1, c2):
    return calcDistance(long_lats[c1], long_lats[c2])

    
def distMatrix(data):
    matrix = []
    for i,d in enumerate(data):
        arr = []
        for j,d1 in enumerate(data):
            if i == j:
                arr.append(0)
            else:
                arr.append(calcDistance(data[i], data[j]))
        matrix.append(arr)
    return matrix

dist_matrix = distMatrix(long_lats)

for i,s in enumerate(states):
    print(i, "  ", s)

start_city = input("Enter start city index: ")
end_city = input("Enter end city index: ")
total_prize_to_collect = input("Enter total prize to collect: ")


import timeit
def Greedy(start, end, prize_collect):
    tot = 0
    tot = tot - prizes[start] - prizes[end]
    visited = []
    currCity = start
    sequence = []
    sequence.append(start)
    dist = 0
    while tot < prize_collect:
        max_ratio = 0
        index = 0
        for i,d in enumerate(long_lats):
            if i in visited or i == end or i == start or i == currCity:
                continue
            ratio = prizes[i]/(dist_matrix[i][currCity])
            if ratio > max_ratio:
                max_ratio = ratio
                index = i
        dist = dist + dist_matrix[currCity][index]
        currCity = index
        visited.append(currCity)
        tot = tot + prizes[index]
        sequence.append(currCity)
    
    sequence.append(end)
    
    et = time.time()
    print("Total Prize Collected: ", tot)
    print("Sequence: ", sequence)
    print("Distance: ", dist)


print("\nRunning Greedy Algorithm")
start = timeit.default_timer()

Greedy(int(start_city), int(end_city), int(total_prize_to_collect))
stop = timeit.default_timer()
print('Time of execution: ', stop - start) 
    
    
import random
import timeit
def RandomAlgorithm(start, end, prize_collect):
    tot = 0
    tot = tot - prizes[start] - prizes[end]
    visited = []
    currCity = start
    sequence = []
    sequence.append(start)
    n = len(prizes)
    dist = 0
    while tot < prize_collect:
        index = random.randint(0,n-1)
        if index not in visited:
            dist = dist + dist_matrix[currCity][index]
            currCity = index
            visited.append(currCity)
            tot = tot + prizes[index]
            sequence.append(currCity)
    
    sequence.append(end)
    print("Total Prize Collected: ", tot)
    print("Sequence: ", sequence)
    print("Distance: ", dist)

print("\nRunning Random Algorithm")  
start = timeit.default_timer()
    
RandomAlgorithm(10,15,100)
stop = timeit.default_timer()
print('Time of execution: ', stop - start) 


import random
import math

def RL(start, end, prize_collect):
    
    #Initialize variables
    num_agents = 10
    visited = []
    paths = []
    lengths = []
    curr_nodes = []
    tot_prizes = []
    best_agent = 0
    for n in range(num_agents):
        visited.append([])
        paths.append([start])
        lengths.append(0)
        curr_nodes.append(start)
        tot_prizes.append(0)
        
    n = len(prizes)
    
    #Initialize Q table and reward table
    Q_table = []
    Reward_table = []
    sum_dist = 0
    for i in range(n):
        for j in range(n):
            sum_dist = sum_dist + dist_matrix[i][j]
    for i in range(n):
        arr = []
        arr2 = []
        for j in range(n):
            val = (n*n)/(n*sum_dist)
            arr.append(val)
            arr2.append(0)
        Q_table.append(arr)
        Reward_table.append(arr2)
    
    learning_rate = 0.1
    discount = 0.3
    alpha = 1.1
    beta = 1.1
    w = 10
    prob_q = 5
    
    #Run algorithm for 100 iterations
    for it in range(100):
        for i in range(n):
            #Select next action for all the agents
            for m in range(num_agents):
                if tot_prizes[m] > prize_collect:
                    continue
                    
                currCity = curr_nodes[m]
                q = random.randint(0,10)
                index = 0
                #Either explore or exploit with probability 0.5
                if q < prob_q:
                    index = 0
                    max_t = 0
                    #Select best state next
                    for c in range(n):
                        if c not in visited[m] and c != currCity and c != start and c != end:
                            ratio = pow(Q_table[c][currCity],alpha)/pow(dist_matrix[c][currCity],beta)
                            if ratio > max_t:
                                max_t = ratio
                                index = c
                else:
                    sum_probs = 0
                    probs = []
                    probs_indices =[]
                    #Select next state according to uniformly distributed weighted probabilities
                    for c in range(n):
                        if c not in visited[m] and c != currCity and c != start and c != end:
                            ratio = pow(Q_table[c][currCity],alpha)/pow(dist_matrix[c][currCity],beta)
                            probs.append(ratio)
                            probs_indices.append(c)
                            sum_probs = sum_probs + ratio
                    for i,p in enumerate(probs):
                        probs[i] = p/sum_probs
                        
                    index = random.choices(population=probs_indices, weights=probs,k=1)[0]
                
                
                visited[m].append(currCity)   
                paths[m].append(index)
                curr_nodes[m] = index
                tot_prizes[m] = tot_prizes[m] + prizes[index]
                lengths[m] = lengths[m] + dist_matrix[index][currCity]
                #Update Q table
                val = (1-learning_rate)*Q_table[index][currCity]
                max_t = 0
                for c in range(n):
                    if c not in visited[m] and c != index and c != start and c != end:
                            if Q_table[c][index] > max_t:
                                max_t = Q_table[c][index]
                val = val + learning_rate*discount*max_t
                Q_table[currCity][index] = val
                
        #Find best agent
        max_t = 0
        index = 0
        for m in range(num_agents):
            ratio = tot_prizes[m]/lengths[m]
            if ratio > max_t:
                max_t = ratio
                index = m
        prev = start
        best_agent = index
        for p in paths[index]:
            #Update reward table
            Reward_table[start][p] = tot_prizes[index]/lengths[index]
            val = Q_table[start][p]*(1-learning_rate)
            max_t = 0
            for i in range(n):
                if i != p:
                    if Q_table[p][i] > max_t:
                        max_t = Q_table[p][i]
            val = val + learning_rate*(Reward_table[start][p] + max_t)
            Q_table[start][p] = val
            start = p
        for n in range(num_agents):
            visited.append([])
            paths.append([start])
            lengths.append(0)
            curr_nodes.append(start)
            tot_prizes.append(0)
    
    paths[best_agent].append(end)
    lengths[best_agent] = lengths[best_agent] + dist_matrix[curr_nodes[best_agent]][end]
    print("Total Prize Collected: ", tot_prizes[best_agent])
    print("Sequence: ", paths[best_agent])
    print("Distance: ", lengths[best_agent])
    

print("\nRunning Deep Reinforcement Learning Algorithm")
start = timeit.default_timer()
RL(10,15,100)         
stop = timeit.default_timer()
print('Time of execution: ', stop - start)               

