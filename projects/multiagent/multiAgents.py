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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        new_ghost_position = [ghost.getPosition() for ghost in newGhostStates]
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        score = successorGameState.getScore()

        "*** YOUR CODE HERE ***"
        # prevents ghost from sitting at one point
        if action == 'Stop':
            return -float('inf')

        # determine if ghost around pacman is scared. if scared, don't runaway. if not, runaway.
        for new_ghost_pos, ghost_scared_time in zip(new_ghost_position, newScaredTimes):
            if newPos == new_ghost_pos and ghost_scared_time <= 0:
                return -float('inf')

        # saving the as the form of {(x, y): distance}
        super_pellets = {loc: util.manhattanDistance(newPos, loc) for loc in successorGameState.getCapsules()}
        normal_pellets = {loc: util.manhattanDistance(newPos, loc) for loc in successorGameState.getFood().asList()}
        ghosts = {loc: util.manhattanDistance(newPos, loc) for loc in successorGameState.getGhostPositions()}

        # declaring default distance dictionary and score
        distance_dic = {'sp': 0, 'np': 0, 'gs': 0}

        # custom setting for numerator as game result was heavily influenced by the numerator setting.
        if super_pellets:
            sp_xy = min(super_pellets, key=super_pellets.get)
            distance_dic['sp'] = float(5 / super_pellets[sp_xy])

        if normal_pellets:
            np_xy = min(normal_pellets, key=normal_pellets.get)
            distance_dic['np'] = float(15 / normal_pellets[np_xy])

        if ghosts:
            gs_xy = min(ghosts, key=ghosts.get)
            distance_dic['gs'] = float(1 / ghosts[gs_xy])

        # history tracking for verbose
        track_history(successorGameState, distance_dic, normal_pellets, score)

        # high score = less food left = good heuristic
        return distance_dic['np'] + distance_dic['sp'] + distance_dic['gs'] + score


def track_history(gs, ds, foods, score):
    print('\
**********************************\n\
remaining foods:\n\
{0}\n\
closest_super_pellet:\n\
{1}\n\
closest_pellet:\n\
{2}\n\
foods: \n\
{3}\n\
heuristic:\n\
{4}\n\
**********************************\n\n'.format(len(gs.getFood().asList()), ds['sp'], ds['np'], foods,
                                               ds['np'] + ds['sp'] + ds['gs'] + score))


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        def maximizer(state, depth):
            if state.isLose() or state.isWin() or self.depth == depth:
                return 0, self.evaluationFunction(state)

            max_dic = {}
            for successor, ac in list((state.generateSuccessor(0, ac), ac) for ac in state.getLegalActions(0)):
                max_dic[calc_minmax(successor, 1, depth)] = ac

            max_value = max(max_dic)
            max_action = max_dic[max_value]
            return max_value, max_action

        def minimizer(state, agent, depth):
            if state.isLose() or state.isWin() or self.depth == depth:
                return 0, self.evaluationFunction(state)

            min_dic = {}
            for successor, ac in list((state.generateSuccessor(agent, ac), ac) for ac in state.getLegalActions(agent)):
                if agent == gameState.getNumAgents() - 1:
                    min_dic[calc_minmax(successor, 0, depth + 1)] = ac
                else:
                    min_dic[calc_minmax(successor, agent + 1, depth)] = ac

            min_value = min(min_dic)
            min_action = min_dic[min_value]
            return min_value, min_action

        # starter
        def calc_minmax(state, agent, depth):
            # Case 1: Pacman (maximizer)
            if agent == 0:
                return maximizer(state, depth)

            # Case 2: Ghost (minimizer)
            return minimizer(state, agent, depth)

        return calc_minmax(gameState, 0, 0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


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
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
