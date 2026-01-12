import os
import json
import pickle
import argparse
from cprint import *

import env.global_vars
from env.graph import Graph
from utils import count_gobal

from instruction import BASE_SYS_PROMPT as SYS_PROMPT

from env.item import Item
from env.tool import Tool
import re
from tqdm import tqdm
import copy
import numpy as np
from typing import List, Tuple
from datetime import datetime

def softmax(a, T=1):
    a = np.array(a) / T
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def copy_state(graph, key_log):
    graph_save = copy.deepcopy(graph)
    bag_save = copy.deepcopy(env.global_vars.bag)
    key_log_save = copy.deepcopy(key_log)
    saved = {"graph_save": graph_save, "bag_save": bag_save, "key_log_save": key_log_save}
    return saved

def get_obs_and_valid_actions(graph: Graph):
    graph.describe()  # 需要 describe() 來更新可用的 action
    interactable_items = []
    adjacent_scenes = []
    tools_in_bag = []
    current_scene = graph.scenes[graph.current_scene]
    for name, item_wrap in current_scene.items.items():
        item: Item = item_wrap["item"]
        if item.visible and item.interactable:
            interactable_items.append((name, item.current_desc()))
    for name, tool_wrap in current_scene.tools.items():
        tool: Tool = tool_wrap["tool"]
        if tool.visible:
            interactable_items.append((name, tool.current_desc()))
    for passage, name in current_scene.scene_relations.items():
        if current_scene.parent_graph.scenes[name].visible:
            adjacent_scenes.append(passage)
    for name, tool in env.global_vars.bag.tools.items():
        tools_in_bag.append((name, tool.describe()))
    obs = {"interactable_items": interactable_items, "adjacent_scenes": adjacent_scenes, "tools_in_bag": tools_in_bag}

    index = 0
    valid_actions = []
    for item in interactable_items:
        # print(index, f": click({item[0]})", sep="")
        valid_actions.append(f"click({item[0]})")
        index += 1
    for tool in tools_in_bag:
        for item in interactable_items:
            # print(index, f": apply({tool[0]}, {item[0]})", sep="")
            valid_actions.append(f"apply({tool[0]}, {item[0]})")
            index += 1
    if(len(tools_in_bag) >= 2):
        for i in range(len(tools_in_bag)-1):
            for j in range(i+1, len(tools_in_bag)):
                # print(index, f": craft({tools_in_bag[i][0]}, {tools_in_bag[j][0]})", sep="")
                valid_actions.append(f"craft({tools_in_bag[i][0]}, {tools_in_bag[j][0]})")
                index += 1
    for passage in adjacent_scenes:
        # print(index, f": move({passage})", sep="")
        valid_actions.append(f"move({passage})")
        index += 1

    return obs, valid_actions

class StateNode:
    def __init__(self, reward=0, done=False):
        self.state = None
        self.prev_state = None
        self.prev_action = None
        self.id = None
        self.valid_actions = None
        # self.action_probs = deque(maxlen=3) # TODO: 刪掉

        self.N = 0
        self.children: List[ActionNode] = []
        self.reward = reward
        self.score = 0
        self.done = done

class ActionNode:
    def __init__(self, action):
        self.action = action
        self.N = 0
        self.Q = 0
        self.Rs = []
        self.children = []
        self.children_text = []

class MCTSAgent:
    def __init__(self):
        self.exploration_constant = 50
        self.max_depth = 15
        self.simulation_per_act = 50
        self.discount_factor = 0.95



    def build_state(self, graph: Graph, key_log, reward=0, done=False):

        state = StateNode()

        obs, valid_actions = get_obs_and_valid_actions(graph)
        
        state.state = obs
        state.reward = reward 

        state.id = str(obs)

        if not done and "input(" in key_log[0][1]:
            state.valid_actions = key_log[0]
            state.children.append(ActionNode(key_log[0]))
        else:
            state.valid_actions = valid_actions
            for valid_action in valid_actions:
                state.children.append(ActionNode(valid_action))

        state.done = done 


        return state

    def search(self, graph: Graph, key_log, log_f) -> Tuple[StateNode, str]:  
        
        max_depth = self.max_depth

        self.root = self.build_state(graph, key_log)

        for _ in tqdm(range(self.simulation_per_act * len(self.root.children))):
        # for _ in tqdm(range(100)):
            saved = copy_state(graph, key_log)
            self.simulate(self.root, graph, key_log, 0, max_depth)
            graph = saved["graph_save"]
            env.global_vars.bag = saved["bag_save"]
            key_log = saved["key_log_save"]

        best_action_node = self.greedy_action_node_UCT(self.root, 0)

        return self.root, best_action_node.action

    def simulate(self, state_node: StateNode, graph: Graph, key_log, depth, max_depth):
        if state_node.done or depth == max_depth:
            return 0
        
        if isinstance(state_node.valid_actions, tuple):
            rollout_next = False

            graph.current_scene = state_node.valid_actions[0]
            graph.describe()
            current_scene_str = graph.current_scene
            action = state_node.valid_actions[1]
            response = graph.react(action)

            best_action_node = state_node.children[0]

        else:
            best_action_node = self.greedy_action_node_UCT(state_node, self.exploration_constant)
            # choose best action based on ucb_value

            rollout_next = False

            graph.describe()  # 需要 describe() 來更新可用的 action
            current_scene_str = graph.current_scene
            action = best_action_node.action
            response = graph.react(action)
            
        reward = 0
        if "move(" not in action and "nothing happen" not in response.lower():
            if "craft(" in action:
                exist = False
                for i, (position, _action) in enumerate(key_log):
                    if _action == action:
                        del key_log[i]
                        exist = True
                        break 
                if not exist:
                    pattern = r"craft\((.*?),\s*(.*?)\)"
                    action_tmp = re.sub(pattern, r"craft(\2, \1)", action)
                    for i, (position, _action) in enumerate(key_log):
                        if _action == action_tmp:
                            del key_log[i]
                            exist = True
                            break 
                if not exist:
                    cprint.fatal(f"{best_action_node.action} not found in key_log")
            else:
                if isinstance(state_node.valid_actions, tuple):
                    key_log.remove(state_node.valid_actions)
                else:
                    key_log.remove((current_scene_str, best_action_node.action))
            reward = 1
        done = False
        if "game end" in response.lower():
            done = True
        obs, _ = get_obs_and_valid_actions(graph)
        next_state_text = str(obs)

        if next_state_text in best_action_node.children_text:
            index = best_action_node.children_text.index(next_state_text)
            next_state_node: StateNode = best_action_node.children[index]

            if next_state_node.N == 0:
                rollout_next = True
            next_state_node.N += 1

        else:
            next_state_node = self.build_state(graph, key_log, reward, done)
            best_action_node.children.append(next_state_node)
            best_action_node.children_text.append(next_state_text)
            rollout_next = True

        # if the next_state.N is 0, then rollout the next state
        # else: simulate the next state
        # recursively rollout/simulate (future * 0.95(discount_factor))
        if rollout_next:
            R = reward + self.discount_factor * self.rollout(next_state_node, graph, key_log, depth+1, max_depth)
        else:
            R = reward + self.discount_factor * self.simulate(next_state_node, graph, key_log, depth+1, max_depth)

        state_node.N += 1
        best_action_node.N += 1

        best_action_node.Rs.append(R)
        best_action_node.Q = np.sum(np.array(best_action_node.Rs) * softmax(best_action_node.Rs, T=10))

        return R
    
    def greedy_action_node_PUCT(self, state_node, exploration_constant):
        # TODO
        pass

    def greedy_action_node_UCT(self, state_node: StateNode, exploration_constant) -> ActionNode:
        best_value = -float('inf')
        best_children = []
        
        for i, child in enumerate(state_node.children):
            # 選出最好的 ucb_value of actions = action的Q + const  * sqrt( ln(state的N) / (action的N) )
            if child.N == 0:
                ucb_value = float('inf')
            else:
                ucb_value = child.Q + exploration_constant * np.sqrt(np.log(state_node.N + 1) / (child.N))

            if np.isclose(ucb_value, best_value):
                best_children.append(child)
            elif ucb_value > best_value:
                best_value = ucb_value
                best_children = [child]

        # return best action based on highest ucb_value
        return np.random.choice(best_children)
    
    def rollout(self, state_node: StateNode, graph: Graph, key_log, depth, max_depth):
        '''
        Randomly choose next action until done or max_depth
        '''
        if state_node.done or depth == max_depth:
            return 0
        
        if isinstance(state_node.valid_actions, tuple):
            graph.current_scene = state_node.valid_actions[0]
            graph.describe()
            current_scene_str = graph.current_scene
            action = state_node.valid_actions[1]
            response = graph.react(action)
            
            action_node = state_node.children[0]
        else:
        # randomly choose an action
            action_node: ActionNode = np.random.choice(state_node.children, 1)[0]
        
            graph.describe()  # 需要 describe() 來更新可用的 action
            current_scene_str = graph.current_scene
            action = action_node.action
            response = graph.react(action)
        reward = 0
        if "move(" not in action and "nothing happen" not in response.lower():
            if "craft(" in action:
                exist = False
                for i, (position, _action) in enumerate(key_log):
                    if _action == action:
                        del key_log[i]
                        exist = True
                        break 
                if not exist:
                    pattern = r"craft\((.*?),\s*(.*?)\)"
                    action_tmp = re.sub(pattern, r"craft(\2, \1)", action)
                    for i, (position, _action) in enumerate(key_log):
                        if _action == action_tmp:
                            del key_log[i]
                            exist = True
                            break 
                if not exist:
                    cprint.fatal(f"{action} not found in key_log")
            else:
                if isinstance(state_node.valid_actions, tuple):
                    key_log.remove(state_node.valid_actions)
                else:
                    # print(response, flush=True)
                    # print(key_log, flush=True)
                    # print((current_scene_str, action), flush=True)
                    key_log.remove((current_scene_str, action))
            reward = 1
        done = False
        if "game end" in response.lower():
            done = True
        obs, _ = get_obs_and_valid_actions(graph)
        next_state_text = str(obs)

        if next_state_text in action_node.children_text:
            index = action_node.children_text.index(next_state_text)
            next_state_node = action_node.children[index]
        else:
            next_state_node = self.build_state(graph, key_log, reward, done)
            action_node.children.append(next_state_node)
            action_node.children_text.append(next_state_text)

        return reward + self.discount_factor * self.rollout(next_state_node, graph, key_log, depth+1, max_depth)

def parse_key_log(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        json_matches = re.findall(r'\{.*?\}', content, re.DOTALL)
        parsed_list = []
        
        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                
                raw_position = data.get("position", "")
                action_answer = data.get("action_answer", "")
                
                clean_position = raw_position.split(" -> ")[-1]
                
                parsed_list.append((clean_position, action_answer))
                
            except json.JSONDecodeError:
                print("json parse error")
                continue
                
        return parsed_list

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []

def make_no_index(response, object_dic, tool_dic):
    try:
        action = response.split("(")[0].strip()
        args = response.split("(")[1].split(")")[0]
        args = args.split(",")
        args = [a.strip() for a in args if a.strip() != ""]
        
        for id, val in enumerate(args):
            if (action == "apply" and id == 0) or action == "craft":
                args[id] = tool_dic[(int)(val)][1]
            elif action == "input" and id == 0:
                pass
            else:
                args[id] = object_dic[(int)(val)][1]
        return action + '(' + ', '.join(args) + ')'
    except:
        return response

if __name__ == "__main__":

    # tmp1 = {"interactable_items": [('telephone cabinet', 'An old-fashioned telephone cabinet stands here.'), ('left door to the study', 'A locked door that leads to the study.'), ('right door to the exit', 'A locked door that leads to the exit, the lock needs a 9-digit code all in numbers')], "adjacent_scenes": ['To the dining room and kitchen', 'To the bedroom'], "tools_in_bag": []}
    # tmp2 = {"interactable_items": [('telephone cabinet', 'An old-fashioned telephone cabinet stands here.'), ('left door to the study', 'A locked door that leads to the study.'), ('right door to the exit', 'A locked door that leads to the exit, the lock needs a 9-digit code all in numbers')], "adjacent_scenes": ['To the dining room and kitchen', 'To the bedroom'], "tools_in_bag": []}
    # print(str(tmp1) == str(tmp2))


    game_index = "1-1"
    game_path = os.path.join("..", "data", f"game{game_index}.yaml")
    keylog_path = os.path.join("..", "data", "reference", f"key_log_{game_index}.txt")
    # graph = Graph(path, use_index=True) # 如果不用 use_index=True scene_prompt 就沒有 index 
    graph = Graph(game_path)
    key_log = parse_key_log(keylog_path)
    env.global_vars.reset_global_vars()

    # print(key_log)

    # cprint.warn(SYS_PROMPT)
    # print()

    agent = MCTSAgent()

    accumulated_reward = 0

    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    with open(os.path.join("log", f"game{game_index}_MCTS_{formatted_time}"), "w") as log_f:
        for step in range(0, 100):

            print("Step:", step+1)
            print("Step:", step+1, file=log_f)
            if "input(" in key_log[0][1]:
                action = key_log[0][1]
                graph.current_scene = key_log[0][0]
            else:
                saved = copy_state(graph, key_log)
                root_node, action = agent.search(graph, key_log, log_f)
                graph = saved["graph_save"]
                env.global_vars.bag = saved["bag_save"]
                key_log = saved["key_log_save"]
                for actionNode in root_node.children:
                    print(actionNode.action, f"Q={actionNode.Q}, Count={actionNode.N}")
                    print(actionNode.action, f"Q={actionNode.Q}, Count={actionNode.N}", file=log_f)
                
            print("Action:", action)
            print("Action:", action, file=log_f)
            current_scene_str = graph.current_scene
            graph.describe()
            response = graph.react(action)
            print("Reward:", response)
            print("Reward:", response, file=log_f)

            if "move(" not in action and "nothing happen" not in response.lower():
                if "craft(" in action:
                    exist = False
                    for i, (position, _action) in enumerate(key_log):
                        if _action == action:
                            del key_log[i]
                            exist = True
                            break 
                    if not exist:
                        pattern = r"craft\((.*?),\s*(.*?)\)"
                        action_tmp = re.sub(pattern, r"craft(\2, \1)", action)
                        for i, (position, _action) in enumerate(key_log):
                            if _action == action_tmp:
                                del key_log[i]
                                exist = True
                                break 
                    if not exist:
                        cprint.fatal(f"{action} not found in key_log")
                else:
                    key_log.remove((current_scene_str, action))
                accumulated_reward += 1
            
            print("Accumulated_reward:", accumulated_reward)
            print("Accumulated_reward:", accumulated_reward, file=log_f)

            if "game end" in response.lower():
                break

            print(file=log_f, flush=True)







