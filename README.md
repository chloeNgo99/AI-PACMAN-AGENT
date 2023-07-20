# AI-PACMAN-AGEN Specification
## Part 1
Apply various search algorithms to aid Pacman in navigating its maze environment. The implemented search algorithms encompass depth-first search (DFS), breadth-first search (BFS), uniform-cost search (UCS), and A* search. By incorporating these algorithms, Pacman will gain the ability to efficiently explore the maze, identifying optimal paths to its destinations.

### Question 1: Finding a Fixed Food Dot using Depth First Search
Implement the depth-first search (DFS) algorithm in the depthFirstSearch function in **search.py.**
Use the graph search version of DFS, which avoids expanding any already visited states.

Test the implementation by running the command:
```
python pacman.py -l tinyMaze -p SearchAgent -a fn=tinyMazeSearch
```
### Question 2: Breadth First Search
Implement the breadth-first search (BFS) algorithm in the breadthFirstSearch function in **search.py.**
Use the graph search version of BFS, which avoids expanding any already visited states.

Test the implementation by running the command:
```
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
```
### Question 3: Varying the Cost Function

Implement the uniform-cost graph search algorithm in the uniformCostSearch function in **search.py**.

While BFS will find a fewest-actions path to the goal, we might want to find paths that are "best" in other senses. Consider mediumDottedMaze and mediumScaryMaze.

By changing the cost function, we can encourage Pacman to find different paths. For example, we can charge more for dangerous steps in ghost-ridden areas or less for steps in food-rich areas, and a rational Pacman agent should adjust its behavior in response.

Test the implementation by running the commands:
```
python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
python pacman.py -l mediumDottedMaze -p StayEastSearchAgent
python pacman.py -l mediumScaryMaze -p StayWestSearchAgent
```
### Question 4: A* Search
Implement A* graph search in the aStarSearch function in **search.py**. A* takes a heuristic function as an argument.

Test the implementation using the Manhattan distance heuristic by running the command:
```
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```
### Question 5: Finding All the Corners
Implement the CornersProblem search problem in **searchAgents.py**.
Choose a state representation that encodes all the information necessary to detect whether all four corners have been reached.

Test the implementation using BFS by running the commands:
```
python pacman.py -l tinyCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
python pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
```
### Question 6: Corners Problem Heuristic

Implement a non-trivial, consistent heuristic for the CornersProblem in the cornersHeuristic function.

Test the implementation using A* search by running the command:
```
python pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5
```
### Question 7: Eating All The Dots
Implement the FoodSearchProblem in **searchAgents.py**. This problem definition formalizes the food-clearing problem.

Test the implementation using BFS and A* search by running the commands:
```
python pacman.py -l testSearch -p AStarFoodSearchAgent
python pacman.py -l trickySearch -p AStarFoodSearchAge
```
## Part 2: Multi-Agent Search
The objective of this project is to design intelligent agents for the classic version of Pacman, encompassing both Pacman and the ghosts. The focus is on implementing key search algorithms, such as minimax and expectimax, and applying them to solve multi-agent scenarios. The goal is to coordinate multiple Pacman agents and ghosts effectively, enhancing the gameplay experience.

### Question 1: Reflex Agent
Improve the **ReflexAgent** in **multiAgents.py** to play respectably. A capable reflex agent will have to consider both food locations and ghost locations to perform well. 

The agent should easily and reliably clear the testClassic layout:
```
python pacman.py -p ReflexAgent -l testClassic
```
Try out the reflex agent on the default **mediumClassic** layout with one ghost or two (and animation off to speed up the display):
```
python pacman.py --frameTime 0 -p ReflexAgent -k 1
python pacman.py --frameTime 0 -p ReflexAgent -k 2
```
You can try the agent out under these conditions with
```
python autograder.py -q q1
```
To run it without graphics, use:
```
python autograder.py -q q1 --no-graphics
```
### Question 2: Minimax
Write an adversarial search agent in the provided **MinimaxAgent** class stub in **multiAgents.py**. The minimax agent should work with any number of ghosts. In particular, the minimax tree will have multiple min layers (one for each ghost) for every max layer.

The code should also expand the game tree to an arbitrary depth. Score the leaves of the minimax tree with the supplied **self.evaluationFunction**, which defaults to **scoreEvaluationFunction**. **MinimaxAgent** extends **MultiAgentSearchAgent**, which gives access to self.depth and **self.evaluationFunction**. Make sure the minimax code makes reference to these two variables where appropriate as these variables are populated in response to command line options.

Important: A single search ply is considered to be one Pacman move and all the ghosts' responses, so depth 2 search will involve Pacman and each ghost moving two times.

To test and debug the code, run
```
python autograder.py -q q2
```
### Question 3: Alpha-Beta Pruning
Make a new agent that uses alpha-beta pruning to more efficiently explore the minimax tree, in **AlphaBetaAgent**, and extend the alpha-beta pruning logic appropriately to multiple minimizer agents.
Test the implementation by running the command:
```
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
```
The **AlphaBetaAgent** minimax values should be identical to the **MinimaxAgent** minimax values, although the actions it selects can vary because of different tie-breaking behavior. Again, the minimax values of the initial state in the **minimaxClassic** layout are 9, 8, 7 and -492 for depths 1, 2, 3 and 4 respectively.

To test and debug the code, run:
```
python autograder.py -q q3
```
### Question 4: Expectimax
Minimax and alpha-beta are great, but they both assume that you are playing against an adversary who makes optimal decisions. As anyone who has ever won tic-tac-toe can tell you, this is not always the case. In this question you will implement the ExpectimaxAgent, which is useful for modeling probabilistic behavior of agents who may make suboptimal choices.

Once the algorithm is working on small trees, you can observe its success in Pacman. Random ghosts are of course not optimal minimax agents, and so modeling them with minimax search may not be appropriate. **ExpectimaxAgent**, will no longer take the min over all ghost actions, but the expectation according to the agent's model of how the ghosts act. To simplify the code, only be running against an adversary which chooses amongst their **getLegalActions** uniformly at random.

To see how the ExpectimaxAgent behaves in Pacman, run:
```
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
```

Now, observe a more cavalier approach in close quarters with ghosts. In particular, if Pacman perceives that he could be trapped but might escape to grab a few more pieces of food, he'll at least try. Investigate the results of these two scenarios:
```
python pacman.py -p AlphaBetaAgent -l trappedClassic -a depth=3 -q -n 10
python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3 -q -n 10
```
The ExpectimaxAgent wins about half the time, while the AlphaBetaAgent always loses. Make sure you understand why the behavior here differs from the minimax case.
### Question 5: Evaluation Function
Write a better evaluation function for pacman in the provided function **betterEvaluationFunction**. The evaluation function should evaluate states, rather than actions like the reflex agent evaluation function did. This may use any tools at the disposal for evaluation, including the search code from the last project. With depth 2 search, the evaluation function should clear the **smallClassic** layout with one random ghost more than half the time and still run at a reasonable rate (to get full credit, Pacman should be averaging around 1000 points when he's winning).
```
python autograder.py -q q5
```
## Part 3: Reinforcement Learning
In this part of the Pacman project, you will explore the field of reinforcement learning to train Pacman agents. Reinforcement learning allows agents to learn through interaction with an environment and receive rewards or penalties based on their actions.

To apply reinforcement learning to Pacman, you can use techniques such as Q-learning or deep Q-networks (DQNs).

### Question 1: Value Iteration
The value iteration equation is as follows:


Write a value iteration agent in **ValueIterationAgent**, which has been partially specified in **valueIterationAgents.py**. The value iteration agent is an offline planner, not a reinforcement learning agent, and so the relevant training option is the number of iterations of value iteration it should run (option -i) in its initial planning phase. **ValueIterationAgent** takes an MDP on construction and runs value iteration for the specified number of iterations before the constructor returns.

Value iteration computes k-step estimates of the optimal values, Vk. In addition to running value iteration, implement the following methods for **ValueIterationAgent** using Vk.

To test the implementation, run the autograder:
```
python autograder.py -q q1
```
The following command loads the **ValueIterationAgent**, which will compute a policy and execute it 10 times. Press a key to cycle through values, Q-values, and the simulation. You should find that the value of the start state (V(start), which you can read off of the GUI) and the empirical resulting average reward (printed after the 10 rounds of execution finish) are quite close.
```
python gridworld.py -a value -i 100 -k 10
```
### Question 2: Bridge Crossing Analysis
**BridgeGrid** is a grid world map with the a low-reward terminal state and a high-reward terminal state separated by a narrow "bridge", on either side of which is a chasm of high negative reward. The agent starts near the low-reward state. With the default discount of 0.9 and the default noise of 0.2, the optimal policy does not cross the bridge. Change only ONE of the discount and noise parameters so that the optimal policy causes the agent to attempt to cross the bridge.
To test the implementation, run the autograder:
```
python autograder.py -q q2
```
### Question 3: Policies
Consider the **DiscountGrid** layout, shown below. This grid has two terminal states with positive payoff (in the middle row), a close exit with payoff +1 and a distant exit with payoff +10. The bottom row of the grid consists of terminal states with negative payoff (shown in red); each state in this "cliff" region has payoff -10. The starting state is the yellow square. We distinguish between two types of paths: (1) paths that "risk the cliff" and travel near the bottom row of the grid; these paths are shorter but risk earning a large negative payoff, and are represented by the red arrow in the figure below. (2) paths that "avoid the cliff" and travel along the top edge of the grid. These paths are longer but are less likely to incur huge negative payoffs. These paths are represented by the green arrow in the figure below.


This question will choose settings of the discount, noise, and living reward parameters for this MDP to produce optimal policies of several different types. The setting of the parameter values for each part should have the property that, if the agent followed its optimal policy without being subject to any noise, it would exhibit the given behavior. If a particular behavior is not achieved for any setting of the parameters, assert that the policy is impossible by returning the string 'NOT POSSIBLE'.

Here are the optimal policy types attempt to produce:

Prefer the close exit (+1), risking the cliff (-10)
Prefer the close exit (+1), but avoiding the cliff (-10)
Prefer the distant exit (+10), risking the cliff (-10)
Prefer the distant exit (+10), avoiding the cliff (-10)
Avoid both exits and the cliff (so an episode should never terminate)
 
To check the answers, run the autograder:
```
python autograder.py -q q3
```
### Question 4: Asynchronous Value Iteration
Write a value iteration agent in **AsynchronousValueIterationAgent** , which has been partially specified in **valueIterationAgents.py**. The value iteration agent is an offline planner, not a reinforcement learning agent, and so the relevant training option is the number of iterations of value iteration it should run (option -i) in its initial planning phase. AsynchronousValueIterationAgent takes an MDP on construction and runs cyclic value iteration (described in the next paragraph) for the specified number of iterations before the constructor returns. Note that all this value iteration code should be placed inside the constructor (__init__ method).

The reason this class is called **AsynchronousValueIterationAgent** is because we will update only one state in each iteration, as opposed to doing a batch-style update. Here is how cyclic value iteration works. In the first iteration, only update the value of the first state in the states list. In the second iteration, only update the value of the second. Keep going until you have updated the value of each state once, then start back at the first state for the subsequent iteration. If the state picked for updating is terminal, nothing happens in that iteration. You can implement it as indexing into the states variable defined in the code skeleton.

As a reminder, here’s the value iteration state update equation:



Value iteration iterates a fixed-point equation, as discussed in class. It is also possible to update the state values in different ways, such as in a random order (i.e., select a state randomly, update its value, and repeat) or in a batch style (as in Q1). In this question, we will explore another technique.

**AsynchronousValueIterationAgent** inherits from **ValueIterationAgent** from Q1, so the only method you need to implement is **runValueIteration**. Since the superclass constructor calls **runValueIteration**, overriding it is sufficient to change the agent’s behavior as desired.

To test the implementation, run the autograder. It should take less than a second to run. If it takes much longer, you may run into issues later in the project, so make the implementation more efficient now.
```
python autograder.py -q q4
```
The following command loads the **AsynchronousValueIterationAgent** in the Gridworld, which will compute a policy and execute it 10 times. Press a key to cycle through values, Q-values, and the simulation. 
```
python gridworld.py -a asynchvalue -i 1000 -k 10
```
### Question 5: Prioritized Sweeping Value Iteration
Implement **PrioritizedSweepingValueIterationAgent**, which has been partially specified in **valueIterationAgents.py**. Note that this class derives from **AsynchronousValueIterationAgent**, so the only method that needs to change is **runValueIteration**, which actually runs the value iteration.

Prioritized sweeping attempts to focus updates of state values in ways that are likely to change the policy.

For this project, we will implement a simplified version of the standard prioritized sweeping algorithm, which is described in this paper Links to an external site.. We’ve adapted this algorithm for our setting. First, we define the predecessors of a state s as all states that have a nonzero probability of reaching s by taking some action a. Also, theta, which is passed in as a parameter, will represent our tolerance for error when deciding whether to update the value of a state. Here’s the algorithm you should follow in the implementation.

1. Compute predecessors of all states.
2. Initialize an empty priority queue.
3. For each non-terminal state s, do the following:
- Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s. This difference represents what the value should be, but the self.values[s] should not be updated in this step.
- Push state s into the priority queue with priority -diff (negative value). This is done because the priority queue is a min heap, but we want to prioritize updating states with a higher error.
4. For each iteration in 0, 1, 2, ..., self.iterations - 1, do the following:
- If the priority queue is empty, terminate.
- Pop a state s off the priority queue.
- Update the value of state s (if it is not a terminal state) in self.values.
- For each predecessor p of s, do the following:
Find the absolute value of the difference between the current value of p in self.values and the highest Q-value across all possible actions from p. This difference represents what the value should be, but the self.values[p] should not be updated in this step. If diff > theta, push state p into the priority queue with priority -diff (negative value), as long as it does not already exist in the priority queue with equal or lower priority. This is done to prioritize updating states with a higher error.

A couple of important notes on implementation:
- When you compute predecessors of a state, make sure to store them in a set, not a list, to avoid duplicates.
- Please use util.PriorityQueue in the implementation. The update method in this class will likely be useful; look at its documentation.

To test the implementation, run the autograder
```
python autograder.py -q q5
```
### Question 6: Q-Learning
Note that the value iteration agent does not actually learn from experience. Rather, it ponders its MDP model to arrive at a complete policy before ever interacting with a real environment. When it does interact with the environment, it simply follows the precomputed policy (e.g. it becomes a reflex agent). This distinction may be subtle in a simulated environment like a Gridword, but it's very important in the real world, where the real MDP is not available.

Write a Q-learning agent, which does very little on construction, but instead learns by trial and error from interactions with the environment through its update(state, action, nextState, reward) method. A stub of a Q-learner is specified in **QLearningAgent** in **qlearningAgents.py**, and you can select it with the option '-a q'. For this question, you must implement the update, **computeValueFromQValues**, **getQValue**, and **computeActionFromQValues** methods.

With the Q-learning update in place, you can watch the Q-learner learn under manual control, using the keyboard:
```
python gridworld.py -a q -k 5 -m
```
Recall that -k will control the number of episodes the agent gets to learn. Watch how the agent learns about the state it was just in, not the one it moves to, and "leaves learning in its wake." Hint: to help with debugging, you can turn off noise by using the --noise 0.0 parameter (though this obviously makes Q-learning less interesting). If you manually steer Pacman north and then east along the optimal path for four episodes, you should see the following Q-values:


To test the implementation, run the autograder
```
python autograder.py -q q6
```
### Question 7: Epsilon Greedy
Complete the Q-learning agent by implementing epsilon-greedy action selection in getAction, meaning it chooses random actions an epsilon fraction of the time, and follows its current best Q-values otherwise. Note that choosing a random action may result in choosing the best action - that is, you should not choose a random sub-optimal action, but rather any random legal action.
```
python gridworld.py -a q -k 100 
```
The final Q-values should resemble those of the value iteration agent, especially along well-traveled paths. However, the average returns will be lower than the Q-values predict because of the random actions and the initial learning phase.

You can choose an element from a list uniformly at random by calling the random.choice function. You can simulate a binary variable with probability p of success by using **util.flipCoin(p)**, which returns True with probability p and False with probability 1-p.

To test the implementation, run the autograder:
```
python autograder.py -q q7
```
With no additional code, you should now be able to run a Q-learning crawler robot:
```
python crawler.py
```
This will invoke the crawling robot from class using the Q-learner. Play around with the various learning parameters to see how they affect the agent's policies and actions. Note that the step delay is a parameter of the simulation, whereas the learning rate and epsilon are parameters of the learning algorithm, and the discount factor is a property of the environment.

### Question 8: Bridge Crossing Revisited
First, train a completely random Q-learner with the default learning rate on the noiseless BridgeGrid for 50 episodes and observe whether it finds the optimal policy.
```
python gridworld.py -a q -k 50 -n 0 -g BridgeGrid -e 1
```
Now try the same experiment with an epsilon of 0. Is there an epsilon and a learning rate for which it is highly likely (greater than 99%) that the optimal policy will be learned after 50 iterations? question8() in analysis.py should return EITHER a 2-item tuple of (epsilon, learning rate) OR the string 'NOT POSSIBLE' if there is none. Epsilon is controlled by -e, learning rate by -l.

Note: the response should be not depend on the exact tie-breaking mechanism used to choose actions. This means the answer should be correct even if for instance we rotated the entire bridge grid world 90 degrees.

To grade the answer, run the autograder:
```
python autograder.py -q q8
```


