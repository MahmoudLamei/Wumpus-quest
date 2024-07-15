"""
    To use this implementation, you simply have to implement `agent_function` such that it returns a legal action.
    You can then let your agent compete on the server by calling
        python3 client_simple.py path/to/your/config.json
    
    The script will keep running forever.
    You can interrupt it at any time.
    The server will remember the actions you have sent.

    Note:
        By default the client bundles multiple requests for efficiency.
        This can complicate debugging.
        You can disable it by setting `single_request=True` in the last line.
"""
import itertools
import json
import logging

import requests
import time

import numpy as np
import pickle



# Returns a dictionary with all the states and their state values.
def value_iteration(states, transitions, rewards, gamma):
    
    # Initiate V(s) = 0
    v = {s: 0 for s in states}
    while True:
        #compute the updated value fn for each state
        new_V = {}
        for s in states:
            values = []
            for a in transitions[s].keys():
                value = 0
                for s2 in transitions[s][a].keys():
                    r = rewards[s][a][s2]
                    t = transitions[s][a][s2]
                    vs2 = v[s2]
                    value += r*t + gamma * t * vs2
                    # print('action:',a,'NextState:',s2,'value:',value)
                
                values.append(value)
                # print('state:',s,'action:',a,'values:', values)
            
            new_V[s] = max(values) if values else 0
        # Check convergence
        if all(abs(v[s] - new_V[s]) < 0.0001 for s in new_V):
            return new_V
        v = new_V


# Removes an element from a tuple by creating another without the element.
def remove(t, e):
    new_tuple = tuple(x for x in t if x != e)
    return new_tuple


# Returns 2 dictionaries with all the transitions and all the rewards
# transitions[((5,9),(),((4,9),))]['NORTH'][((6,9),(),())] = 0.833
# rewards[((5,9),(),((4,9),))]['NORTH'][((6,9),(),())] = 1
def get_transition_reward(states, actions, map_info, skill_points):
    nav = skill_points['navigation']
    fighting = skill_points['fighting']
    accessable = map_info[0]
    stairs = map_info[1]
    m_coins = map_info[2]
    pits = map_info[3]
    teles = map_info[5]

    coin_reward = 1
    moving_reward = -0.01
    win_fight_reward = 0
    success_tele_reward = 0
    exit_reward = 0.01

    transitions = {}
    rewards = {}
    for state in states:
        pos, wumpus, coins, alive = state

        coins_collected = len(m_coins) - len(coins) + 1
        lose_fight_reward = -coins_collected
        fail_tele_reward = -coins_collected
        pit_reward = -coins_collected

        neighbors = get_neighbors(accessable, pos)
        action_transitions = {}
        action_rewards = {}
        free_spots = 4 - list(neighbors.values()).count(None)-1
        for action in actions:  
            next_states = {}
            next_states_rewards = {}
            moving = ['NORTH', 'EAST', 'SOUTH', 'WEST'] 
            # if the action is move and the cell is free.
            if (action in moving) and (pos not in wumpus) and (neighbors[action] != None):
                for action2 in moving:
                    new_pos = pos
                    match action2:
                        case 'NORTH': new_pos = (pos[0]-1, pos[1])
                        case 'EAST': new_pos = (pos[0], pos[1]+1)
                        case 'SOUTH': new_pos = (pos[0]+1, pos[1])
                        case 'WEST': new_pos = (pos[0], pos[1]-1)
                    if new_pos in accessable:
                        if new_pos in coins:
                            coins2 = remove(coins, new_pos)
                            next_state = (new_pos, wumpus, coins2, True)
                            next_state_reward = coin_reward
                        elif new_pos in pits:
                            next_state = (new_pos, wumpus, coins, False)   
                            next_state_reward = pit_reward
                        else:
                            next_state = (new_pos, wumpus, coins, True)
                            next_state_reward = moving_reward

                        if next_state not in states:
                                    states.append(next_state)
                        
                        if action == action2:    
                            if nav == 0:
                                next_states[next_state] = 0
                            else:
                                next_states[next_state] = (nav-1)/nav
                            if free_spots == 0:
                                next_states[next_state] = 1
                            
                        else:
                            if nav == 0:
                                next_states[next_state] = 1 / free_spots
                            else:
                                next_states[next_state] = (1/nav) / free_spots
                        next_states_rewards[next_state] = next_state_reward

            # if the action is fight and there is a wumpus in the position.
            elif (action == 'FIGHT') and (pos in wumpus):
                # Killed the wumpus.
                wumpus2 = remove(wumpus, pos)
                next_state = (pos, wumpus2, coins, True)
                if next_state not in states:
                    states.append(next_state)
                next_states[next_state] = (fighting-2)/fighting if fighting != 0 else 0
                next_states_rewards[next_state] = lose_fight_reward

                # Died by the wumpus.
                next_state = (pos, wumpus, coins, False)
                if next_state not in states:
                    states.append(next_state)
                next_states[next_state] = 2/fighting if fighting != 0 else 1
                next_states_rewards[next_state] = win_fight_reward

            # if the action is teleport and there is a teleport in the position.
            elif (action == 'TELEPORT') and (pos in teles):
                # Successfuly teleported.
                teleports = teles.copy()
                teleports.remove(pos)
                for tele in teleports:
                    next_state = (tele, wumpus, coins, True)
                    if next_state not in states:
                            states.append(next_state)
                    next_states[next_state] = ((nav-1)/nav)/len(teleports) if nav != 0 else 0
                    next_states_rewards[next_state] = success_tele_reward
                # Failed to teleport.
                next_state = (pos, wumpus, coins, False)
                if next_state not in states:
                    states.append(next_state)
                next_states[next_state] = 1/nav if nav != 0 else 1
                next_states_rewards[next_state] = fail_tele_reward

            elif (action == 'EXIT') and (pos in stairs):
                next_state = ((-1,0), wumpus,coins, False)
                if next_state not in states:
                    states.append(next_state)
                next_states[next_state] = 1
                next_states_rewards[next_state] = exit_reward

            action_transitions[action] = next_states
            action_rewards[action] = next_states_rewards
        
        # TOOD: Merge those two for loops.
        for key, value in list(action_transitions.items()):  # Use list() to create a copy of items for safe iteration
            if not value:  # Check if the value is an empty dictionary
                del action_transitions[key]  # Delete the key from the dictionary
        for key, value in list(action_rewards.items()):
            if not value:
                del action_rewards[key]  # Delete the key from the dictionary

        transitions[state] = action_transitions
        rewards[state] = action_rewards
    return transitions, rewards


# Returns a dictionary with the neighbors positions if inside the map, otherwise it's None.
def get_neighbors(acc_pos, loc):
    x, y = loc
    neighbor = {}
    neighbor['NORTH'] = (x-1, y) if (x-1,y) in acc_pos else None
    neighbor['EAST'] = (x, y+1) if (x, y+1) in acc_pos else None
    neighbor['SOUTH'] = (x+1, y) if (x+1, y) in acc_pos else None
    neighbor['WEST'] = (x, y-1) if (x, y-1) in acc_pos else None
    return neighbor


# Returns an array with the positions of all the special blocks in the map.
def get_info(map):
    space = [(x, y) for x, y in np.argwhere(map == ' ')]
    stairs = [(x, y) for x, y in np.argwhere(map == 'S')]
    coins = [(x, y) for x, y in np.argwhere(map == 'G')]
    pit = [(x, y) for x, y in np.argwhere(map == 'P')]
    wumpus = [(x, y) for x, y in np.argwhere(map == 'W')]
    tele = [(x, y) for x, y in np.argwhere(map == 'T')]
    accessable = space + stairs + coins + pit + wumpus + tele
    return [accessable,stairs,coins,pit,wumpus,tele]


# Takes the history from the server to return the current state.
def get_current_state(map_info, history, values):
    accessable, stairs, coins, pit, wumpus, tele = map_info
    pos = stairs[0]
    w = wumpus.copy()
    c = coins.copy()
    for req in history:
        output = req['outcome']
        if 'killed-wumpus-at' in output:
            w.remove((output['killed-wumpus-at'][1],output['killed-wumpus-at'][0]))
        if 'collected-gold-at' in output:
            c.remove((output['collected-gold-at'][1],output['collected-gold-at'][0]))
        if 'position' in output:
            pos = (output['position'][1],output['position'][0])
    current_state = (pos, tuple(w), tuple(c), True)

    for s in values.keys():
        if (s[0] == current_state[0]) and (set(s[1]) == set(current_state[1])) and (set(s[2]) == set(current_state[2]) and (s[3] == current_state[3])):
            current_state = s

    return current_state


def agent_function(request_data):
    # TOOD: Implement this function in a better way
    print('\nI got the following request:')
    print(request_data)

    map_s = request_data['map']
    history = request_data['history']
    actions = ['NORTH', 'EAST', 'SOUTH', 'WEST', 'FIGHT', 'TELEPORT', 'EXIT']
    map = np.array([list(row) for row in map_s.split('\n')])
    map_info = get_info(map)
    accessable, stairs, coins, pit, wumpus, tele = map_info
    if not 'skill-points' in request_data:
        free_points = request_data['free-skill-points']
        if wumpus:
            return {'navigation': int(free_points/2), 'fighting': int(free_points/2)}
        else:
            return {'navigation': free_points, 'fighting': 0}

    else:
        skill_points = request_data['skill-points']


    # READ the pickle file and check if we have the info for this env.
    try:
        with open("data.pickle", "rb") as pickle_file:
            data = pickle.load(pickle_file)
    except EOFError:
        print('Failed to load data')
        data = {}

    
    if map_s not in data.keys():
        states= [(stairs[0], tuple(wumpus), tuple(coins), True)]
        transitions, rewards = get_transition_reward(states, actions, map_info, skill_points)
        gamma = 0.99
        values = value_iteration(states, transitions, rewards, gamma)
        data[map_s] = {'transitions': transitions, 'rewards': rewards, 'values': values}
        print("Used NEW data")
    else:
        transitions = data[map_s]['transitions']
        rewards = data[map_s]['rewards']
        values = data[map_s]['values']
        print('Used SAVED data')

    # If the data file gets too big delete everything and keep the last 5 entries.
    if len(data) > 20:
        data = {key: data[key] for key in list(data.keys())[-5:]}
    # Write the pickle file.
    with open("data.pickle", "wb") as pickle_file:
        pickle.dump(data, pickle_file)

    current_state = get_current_state(map_info, history, values)
    best = -np.inf
    chosen_action = 'NORTH'
    
    for a in transitions[current_state].keys():
        v = 0
        for s2 in transitions[current_state][a].keys():
            v += (values[s2] + rewards[current_state][a][s2]) * transitions[current_state][a][s2]
        if v > best:
            best = v
            chosen_action = a
    if chosen_action == 'EXIT' and map_s in data.keys():
        del data[map_s]
    return chosen_action


def run(config_file, action_function, single_request=False):
    logger = logging.getLogger(__name__)

    with open(config_file, 'r') as fp:
        config = json.load(fp)
    
    logger.info(f'Running agent {config["agent"]} on environment {config["env"]}')
    logger.info(f'Hint: You can see how your agent performs at {config["url"]}agent/{config["env"]}/{config["agent"]}')

    actions = []
    for request_number in itertools.count():
        logger.debug(f'Iteration {request_number} (sending {len(actions)} actions)')
        # send request
        response = requests.put(f'{config["url"]}/act/{config["env"]}', json={
            'agent': config['agent'],
            'pwd': config['pwd'],
            'actions': actions,
            'single_request': single_request,
        })
        if response.status_code == 200:
            response_json = response.json()
            for error in response_json['errors']:
                logger.error(f'Error message from server: {error}')
            for message in response_json['messages']:
                logger.info(f'Message from server: {message}')

            action_requests = response_json['action-requests']
            if not action_requests:
                logger.info('The server has no new action requests - waiting for 1 second.')
                time.sleep(1)  # wait a moment to avoid overloading the server and then try again
            # get actions for next request
            actions = []
            for action_request in action_requests:
                actions.append({'run': action_request['run'], 'action': action_function(action_request['percept'])})
        elif response.status_code == 503:
            logger.warning('Server is busy - retrying in 3 seconds')
            time.sleep(3)  # server is busy - wait a moment and then try again
        else:
            # other errors (e.g. authentication problems) do not benefit from a retry
            logger.error(f'Status code {response.status_code}. Stopping.')
            break


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import sys
    run(sys.argv[1], agent_function, single_request=False)
