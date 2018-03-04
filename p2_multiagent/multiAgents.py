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
import numpy as np

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        #print "successorGameState"
        #print successorGameState
        newPos = successorGameState.getPacmanPosition()		# pos tuple
        newFood = successorGameState.getFood()				# boolean matrix
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        #"*** YOUR CODE HERE ***"
        ghost_dists = [manhattanDistance(pos, newPos) for pos in successorGameState.getGhostPositions()]
        food_dists = [manhattanDistance(pos, newPos) for pos in newFood.asList()]
        
        food_min = 0.0 if len(food_dists)==0 else np.min(food_dists)
        ghost_min = ghost_dists[0] if newScaredTimes[0]==0 else 0.0
        return ghost_min - food_min + successorGameState.getScore()


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
        #"*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        
        def min_max_search(cur_state, depth, agent_id):
            stop_flag = depth == self.depth and agent_id==0
            win_flag = cur_state.isWin()
            lose_flag = cur_state.isLose()
            if stop_flag or win_flag or lose_flag:
                return self.evaluationFunction(cur_state)                  # a value
            
            succ_actions = cur_state.getLegalActions(agent_id)
            total_agents = cur_state.getNumAgents()
            if agent_id == 0:		# pacman's turn
                succ_score = max([min_max_search(cur_state.generateSuccessor(agent_id, act), depth+1, agent_id+1) \
                                    for act in succ_actions])
            else:					# ghost's turn
                succ_score = min([min_max_search(cur_state.generateSuccessor(agent_id, act), depth, (agent_id+1)%total_agents) \
                                    for act in succ_actions])
            return succ_score

        chosen_action = max(gameState.getLegalActions(0), \
        	key=lambda act: min_max_search(gameState.generateSuccessor(0, act), 1, 1))
        return chosen_action



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        #"*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        pos_infty, neg_infty = 10000, -10000
        total_agents = gameState.getNumAgents()

        def max_value(cur_state, depth, agent_id, alpha, beta):
            assert agent_id == 0, "max_value should only be applied on pacman (agent_id=0)"
            if depth > self.depth:
                return self.evaluationFunction(cur_state)
            
            cur_v = None
            next_agent_id = agent_id+1
            for act in cur_state.getLegalActions(agent_id):
                act_score = min_value(cur_state.generateSuccessor(agent_id, act), depth, next_agent_id, alpha, beta)
                cur_v = max(cur_v, act_score)
                if cur_v>beta:
                    return cur_v
                alpha = max(alpha, cur_v)

            if cur_v is None:
                return self.evaluationFunction(cur_state)
            return cur_v

        def min_value(cur_state, depth, agent_id, alpha, beta):
            assert agent_id > 0, "min_value should only be applied on ghost (agent_id>0)"
            
            cur_v = None
            next_agent_id = (agent_id+1)%total_agents
            if next_agent_id == 0:
                min_or_max, depth = max_value, depth+1
            else:
                min_or_max = min_value

            for act in cur_state.getLegalActions(agent_id):
                act_score = min_or_max(cur_state.generateSuccessor(agent_id, act), depth, next_agent_id, alpha, beta)
                if cur_v is None:
                    cur_v = act_score
                else:
                    cur_v = min(cur_v, act_score)
                
                if cur_v<alpha:
                    return cur_v
                beta = min(beta, cur_v)

            if cur_v is None:
                return self.evaluationFunction(cur_state)
            return cur_v


        cur_v, alpha, beta, chosen_action = neg_infty, neg_infty, pos_infty, None
        for act in gameState.getLegalActions(0):
            cur_v = max(cur_v, min_value(gameState.generateSuccessor(0, act), 1, 1, alpha, beta))
            alpha, chosen_action = max(cur_v, alpha), act if cur_v > alpha else chosen_action

        return chosen_action



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
        #"*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def expectimax(cur_state, depth, agent_id):
            stop_flag = depth == self.depth and agent_id==0
            win_flag = cur_state.isWin()
            lose_flag = cur_state.isLose()
            if stop_flag or win_flag or lose_flag:
                return self.evaluationFunction(cur_state)                  # a value
            
            succ_actions = cur_state.getLegalActions(agent_id)
            total_agents = cur_state.getNumAgents()
            if agent_id == 0:		# pacman's turn
                succ_score = max([expectimax(cur_state.generateSuccessor(agent_id, act), depth+1, agent_id+1) \
                                    for act in succ_actions])
            else:					# ghost's turn
                succ_score = 1.0*sum([expectimax(cur_state.generateSuccessor(agent_id, act), depth, (agent_id+1)%total_agents) \
                                    for act in succ_actions]) / len(succ_actions)
            return succ_score

        chosen_action = max(gameState.getLegalActions(0), \
            key=lambda act: expectimax(gameState.generateSuccessor(0, act), 1, 1))
        return chosen_action




def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    #"*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    ''' # This precisely designed method doesn't work well

    newPos = currentGameState.getPacmanPosition()		# pos tuple
    newFood = currentGameState.getFood()				# boolean matrix
    newGhostStates = currentGameState.getGhostStates()
    newCaps = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    ghost_dists = [manhattanDistance(pos, newPos) for pos in currentGameState.getGhostPositions()]
    closeGhost = []
    for i in range(len(ghost_dists)):
    	if newScaredTimes[i]>0:
    		closeGhost.append(20-ghost_dists[i])
        if ghost_dists[i]==1:
            closeGhost.append(-1)  # too close, should avoid!
        elif ghost_dists[i]>=5:
            closeGhost.append(10) # too far, could ignore
        else:
            closeGhost.append(ghost_dists[i])
    ghostSum = sum(closeGhost)

    food_dists = np.sort([manhattanDistance(pos, newPos) for pos in newFood.asList()])
    closeFood_sum = sum(food_dists[:5]) if len(food_dists)>10 \
    				else (np.min(food_dists) if len(food_dists)>0 else 0.0)

    #print len(ghost_dists)
    caps_dist = [manhattanDistance(pos, newPos) for pos in newCaps]
    caps_min = 0.0 if len(caps_dist)==0 else np.min(caps_dist)

    print ghostSum, -closeFood_sum, -10*len(newCaps), -2*caps_min
    return ghostSum - closeFood_sum - len(newCaps) - caps_min
    '''

    #6/6  # This roughly designed method work well...interesting
    newPos = currentGameState.getPacmanPosition()		
    newFood = currentGameState.getFood()				
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    ghost_dists = [manhattanDistance(pos, newPos) for pos in currentGameState.getGhostPositions()]
    food_dists = [manhattanDistance(pos, newPos) for pos in newFood.asList()]

    food_min = 0.0 if len(food_dists)==0 else np.min(food_dists)
    ghost_min = ghost_dists[0] if newScaredTimes[0]==0 else 100.0
	
    return ghost_min - food_min + currentGameState.getScore()
	

# Abbreviation
better = betterEvaluationFunction

