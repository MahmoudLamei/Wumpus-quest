# WS2324 Assignment 2.3 Wumpus Quest


**Topic:** In this assignment we are trying to collect as many gold coins as possible putting in mind the risk of dying due to various elements in the map.

**Dependencies:** You can download all the dependencies by running ```pip install -r requirements.txt```.

**Running:** Run the command ```python3 client_simple.py env-1.json``` you can choose env between 1-5.

# Description
The Wumpus World is a cell-based map where each cell has a special effect on the agent, the agent here is trying to collect coins, and has the map, so that every cell is known but there is noise and there is probability that it would not move in the way it wanted to depending on the skills.

There are two skills, navigation and fighting. Everytime the agent moves it rolls a dice with a number of faces equals to the skill points in navigation and if it gets 2 or more then it does as it wanted, otherwise it goes to a random free cell. The map has some wumpuses and if the agent ends up in their cell, it has to fight by rolling a dice and getting 3 or more. 

For an example the agent wants to move south and it rolled a dice, but it got 1 so it moved north instead, it ended up in a cell with a wumpus so it had to fight and roll another dice, it got a 4 so it killed the wumpus. 

There are also teleports that can be used and will teleport you to another random teleport and also by using the navigation skill, so the agent will also have to roll a dice everytime it teleports.

The map also has pits where the agent simply instantly dies. The agent always starts in a stairs cell and has to go out of the map by using that staircase. You would always start with 1 coin so there is a possibility that the agent will just decide to save the coin it has and not risk it. Also there is a time limit to the problem where you can only take a max of 100 actions or the agent dies.

To build such agent we will use a reinforcement learning approach and use value iteration, where the agent learns about every possible state and gives it a value where it can use that later to pick the action that will get it the best state value. To calculate the state value we will use the Bellman optimality equation. To use this technique we have to assign rewards to every event such as collecting a coin, killing a wumpus, exiting alive and so on. The trouble I faced was with fine tuning the rewards, sometimes the wumpus would decide that it is going to die anyways and just kills itself because it sees that this way it's not going to take a bad penalty as trying to survive, sometimes it would just keep spining around itself till it dies. Finding the perfect rewards and penalties was an obstacle that was passed after a lot of trial and error. 