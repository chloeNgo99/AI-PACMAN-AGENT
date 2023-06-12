# Search
## DFS
In this problem, we used a stack data structure to keep track of the new states and a visited array to mark seen those visited states. We start by adding the start state in the stack, then use a while loop to check whether the element from the stack is a goal state, then return the direction. Otherwise, we continue exploring the neighbors and check if the neighbors have already visited. If it's not, we add to the visited array and push all its successors onto the stack. The algorithm continues until either a goal state is found, in which case it returns the path to the goal state, or the stack becomes empty, in which case it returns an empty list.

## BFS
Literally implement the same algorithm as dfs but we use a queue data structure instead of a stack.

## UCS
For the UCS, we implement a priority queue to sort the distance from the start state to every possible path we can explore, with the smallest distance always appearing on top of the queue. At each iteration, the algorithm removes the node with the lowest path cost from the frontier and checks whether it is a goal node. If it is, it returns the path taken to reach the goal. Otherwise, the algorithm expands the node by generating its successors, updating their path cost and direction, and adding them to the frontier.

## A* SEARCH
The function creates a priority queue fringe and a list seen to keep track of the visited states. The fringe initially contains the start state and its estimated cost to reach the goal state, which is the sum of the cost to reach the start state and the estimated cost to reach the goal state using the given heuristic function. Then, the function enters into a loop that runs until the fringe is not empty. In each iteration, it pops the state with the lowest estimated cost from the fringe. If the popped state is the goal state, it returns the actions taken to reach it. If not, it checks whether the state has already been visited. If not, it adds the state to the seen list, generates its successors, and adds them to the fringe with their estimated cost.


# SearchAgent
## CornersProblem Class
### init function:
    Create an extra visited_corner array to record all corners has visited latter in the program.

### getStartState function:
    Simply returns the start state and not the full Pacman state space

### isGoalState function:
    Check if all visited states cover the corners

### getSuccessors function:
    The method checks each possible direction and determines whether it is legal or not, and whether the new position hits a wall or not. If the new position is a legal move, then it becomes a successor of the current state, with a cost of 1. If the successor state is one of the corners, it is recorded in the visited_corners list. This list represents which corners have been visited, by setting the element corresponding to the visited corner to True. Finally, the method increments the _expanded counter to keep track of the number of nodes that have been expanded in the search

## cornersHeuristic
The heuristic function uses the Manhattan distance as the heuristic value. The function first loops through all corners and stores the unvisited one in the unvisited_corners array. Then we use another loop to loop through each unvisited corner, calculate the Manhattan distance between the current position and the unvisited corner, and then add the result to a list of distances. After the loop is done executing, we then take the min value from distance list and return the result.

## foodHeuristic
The foodHeuristic function loop through each item in the food list and then implements the mazeDistance function to find the optimal distance between the start state to position and the food position. For each iteration, we will only keep the max distance. The reasoning is that if Pacman eats the food items that are farther away first, it will be more efficient than taking a path that leads to closer food items but is longer in the long run.

## ClosestDotSearchAgent Class
### findPathToClosestDot function:
    The function first retrieves the Pacman's current position, the layout of the food, and the walls from the provided gameState. It then creates a new AnyFoodSearchProblem instance with the gameState object to define the problem and pass it to the breadthFirstSearch function from the search module to find a path to the closest food pellet using BFS. After that we just simply return
    the bfs path.

### isGoalState function:
    The isGoalState method should return True if the state represents the goal state of the problem, which is when all the food has been eaten.
