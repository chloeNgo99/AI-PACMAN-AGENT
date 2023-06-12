# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        scareTime = min(newScaredTimes)
    
        if len(foodDistances) > 0:
            foodDistance = min(foodDistances)
            print("min foodDistance: ", foodDistance)
        else:
            foodDistance = 0

        if len(ghostDistances) > 0:
            ghostDistance = min(ghostDistances)
            print("min ghostDistance: ", ghostDistance)
        else:
            ghostDistance = 0

        if ghostDistance == 0:
            ghostDistance = 1
        if foodDistance == 0:
            foodDistance = 1
       
        if scareTime == 0:
            scareTime = 1
       
        return successorGameState.getScore() + 1.0 / foodDistance - 1.0 / ghostDistance - scareTime

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState, agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return max(minimax(gameState.generateSuccessor(agentIndex, action), 1, depth) for action in gameState.getLegalActions(agentIndex))
            else:
                nextAgent = agentIndex + 1
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                return min(minimax(gameState.generateSuccessor(agentIndex, action), nextAgent, depth) for action in gameState.getLegalActions(agentIndex))
        return max(gameState.getLegalActions(0), key=lambda x: minimax(gameState.generateSuccessor(0, x), 1, 0))

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(state, depth, agent, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agent == 0:
                v = float('-inf')
                for action in state.getLegalActions(agent):
                    v = max(v, alphabeta(state.generateSuccessor(agent, action), depth, agent+1, alpha, beta))
                    if v > beta:
                        return v
                    alpha = max(alpha, v)
                return v
            else:
                v = float('inf')
                for action in state.getLegalActions(agent):
                    if agent == gameState.getNumAgents() - 1:
                        v = min(v, alphabeta(state.generateSuccessor(agent, action), depth+1, 0, alpha, beta))
                    else:
                        v = min(v, alphabeta(state.generateSuccessor(agent, action), depth, agent+1, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v
            
        best_score = float('-inf')
        best_action = None
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            v = alphabeta(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            if v > best_score:
                best_score = v
                best_action = action
            alpha = max(alpha, best_score)
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(gameState, agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return max(expectimax(gameState.generateSuccessor(agentIndex, action), 1, depth) for action in gameState.getLegalActions(agentIndex))
            else:
                nextAgent = agentIndex + 1
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                return sum(expectimax(gameState.generateSuccessor(agentIndex, action), nextAgent, depth) for action in gameState.getLegalActions(agentIndex)) / len(gameState.getLegalActions(agentIndex))
        return max(gameState.getLegalActions(0), key=lambda x: expectimax(gameState.generateSuccessor(0, x), 1, 0))

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    
    As for your reflex agent evaluation function, you may want to use the reciprocal of important values 
    (such as distance to food) rather than the values themselves.
    One way you might want to write your evaluation function is to use a linear combination of features. 
    That is, compute values for features about the state that you think are important, and then combine 
    those features by multiplying them by different values and adding the results together. You might 
    decide what to multiply each feature by based on how important you think it is.
    """
    
    "*** YOUR CODE HERE ***"
    # Get the current game state information
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Get the food list
    foodList = food.asList()
    # Get the ghost list
    ghostList = [ghostState.getPosition() for ghostState in ghostStates]
    # Get the capsule list
    capsuleList = capsules

    # Get the number of food
    numFood = len(foodList)
    # Get the number of capsules
    numCapsule = len(capsuleList)
    # Get the number of ghost
    numGhost = len(ghostList)

    # Get the distance to the nearest food
    if numFood > 0:
        nearestFood = min([manhattanDistance(pacmanPos, foodPos) for foodPos in foodList])
    else:
        nearestFood = 0
    # Get the distance to the nearest ghost
    if numGhost > 0:
        nearestGhost = min([manhattanDistance(pacmanPos, ghostPos) for ghostPos in ghostList])
    else:
        nearestGhost = 0
    # Get the distance to the nearest capsule
    if numCapsule > 0:
        nearestCapsule = min([manhattanDistance(pacmanPos, capsulePos) for capsulePos in capsuleList])
    else:
        nearestCapsule = 0

    # Get the reciprocal of the distance to the nearest food
    if nearestFood > 0:
        nearestFood = 1.0 / nearestFood
    # Get the reciprocal of the distance to the nearest ghost
    if nearestGhost > 0:
        nearestGhost = 1.0 / nearestGhost
    # Get the reciprocal of the distance to the nearest capsule
    if nearestCapsule > 0:
        nearestCapsule = 1.0 / nearestCapsule

    # Get the reciprocal of the number of food
    if numFood > 0:
        numFood = 1.0 / numFood
    # Get the reciprocal of the number of capsules
    if numCapsule > 0:
        numCapsule = 1.0 / numCapsule
    # Get the reciprocal of the number of ghost
    if numGhost > 0:
        numGhost = 1.0 / numGhost

    # Get the reciprocal of the scared time of the nearest ghost
    if nearestGhost > 0:
        nearestGhostScaredTime = scaredTimes[ghostList.index(min(ghostList, key=lambda x: manhattanDistance(pacmanPos, x)))]
        if nearestGhostScaredTime > 0:
            nearestGhostScaredTime = 1.0 / nearestGhostScaredTime
    else:
        nearestGhostScaredTime = 0

    return currentGameState.getScore() + nearestFood + nearestGhost + nearestCapsule + numFood + numCapsule + numGhost + nearestGhostScaredTime
    

# Abbreviation
better = betterEvaluationFunction
