def read_input():
    n = int(input())
    start = []
    end = []
    d_time = []  # duration time at each node
    times = []   # travel times between nodes
    for _ in range(n):
        s, e, d = map(int, input().split())
        start.append(s)
        end.append(e)
        d_time.append(d)
    for _ in range(n+1):
        line = list(map(int, input().split()))
        times.append(line)
    return n, start, end, d_time, times

def checkIfValid_Node(node, start, end, arrival_time, d_time):
    # Check if node is unvisited and arrival time is within window
    node_idx = node - 1
    return (arrival_time >= start[node_idx] and 
            arrival_time <= end[node_idx])

def backtrack(cur_step, path, visited, cur_time, start, end, d_time, times, n, best_path, min_time):
    print("cur_step: ",cur_step)
    print("cur_time: ",cur_time)
    # Base case: all nodes visited
    if cur_step == n:
        if cur_time + times[path[cur_step]][0] < min_time:  # Using list to allow modification
            min_time = cur_time  + times[path[cur_step]][0]
            best_path[:] = path[:]
        return
    
    
    prev_node = 0 if cur_step == 1 else path[cur_step-1]
    for node in range(1, n+1):                                                                                           
        if not visited[node]:
            travel_time = times[prev_node][node] 
            arrival_time = cur_time + travel_time
            
            if checkIfValid_Node( node, start, end, arrival_time, d_time):
                print("call valid...")
                visited[node] = True
                path[cur_step] = node
                new_time = arrival_time + d_time[node-1]
                print("cur path:", path)
                backtrack(cur_step + 1, path, visited, new_time, 
                         start, end, d_time, times, n, best_path, min_time)
                
                visited[node] = False
                path[cur_step] = 0

def main():
    n, start, end, d_time, times = read_input()
    visited = [False] * (n + 1)
    path = [0] * (n+1)  
    best_path = [0] * n
    min_time = float('inf')  
    visited[0] = True

    backtrack(1, path, visited, 0, start, end, d_time, times, n, best_path, min_time)
    
    if min_time == float('inf'):
        print("No valid solution found")
    else:
        print(n)
        print(*best_path)

if __name__ == "__main__":
    main()