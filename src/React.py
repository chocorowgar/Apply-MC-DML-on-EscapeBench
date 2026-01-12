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
import time
import openai
from openai import OpenAI
from typing import Mapping
from datetime import datetime
from collections import deque


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
    interactable_items = {}
    to_adjacent_scenes = []
    adjacent_scenes = []
    tools_in_bag = {}
    current_scene = graph.scenes[graph.current_scene]
    for name, item_wrap in current_scene.items.items():
        item: Item = item_wrap["item"]
        if item.visible and item.interactable:
            interactable_items[name] = item.current_desc()
    for name, tool_wrap in current_scene.tools.items():
        tool: Tool = tool_wrap["tool"]
        if tool.visible:
            interactable_items[name] = tool.current_desc()
    for passage, name in current_scene.scene_relations.items():
        if current_scene.parent_graph.scenes[name].visible:
            to_adjacent_scenes.append(passage)
            adjacent_scenes.append(name)
    for name, tool in env.global_vars.bag.tools.items():
        tools_in_bag[name] = tool.describe()
    obs = {"current_scene": graph.current_scene, "interactable_items": interactable_items, "adjacent_scenes": adjacent_scenes, "tools_in_bag": tools_in_bag}

    index = 0
    valid_actions = []
    for item in interactable_items:
        # print(index, f": click({item[0]})", sep="")
        valid_actions.append(f"click({item})")
        index += 1
    for tool in tools_in_bag:
        for item in interactable_items:
            # print(index, f": apply({tool[0]}, {item[0]})", sep="")
            valid_actions.append(f"apply({tool}, {item})")
            index += 1
    if(len(list(tools_in_bag.keys())) >= 2):
        for i in range(len(list(tools_in_bag.keys()))-1):
            for j in range(i+1, len(list(tools_in_bag.keys()))):
                # print(index, f": craft({tools_in_bag[i][0]}, {tools_in_bag[j][0]})", sep="")
                valid_actions.append(f"craft({list(tools_in_bag.keys())[i]}, {list(tools_in_bag.keys())[j]})")
                index += 1
    for passage in to_adjacent_scenes:
        # print(index, f": move({passage})", sep="")
        valid_actions.append(f"move({passage})")
        index += 1

    return obs, valid_actions

def update_memory(memory, obs, investigated = None):
    if obs["current_scene"] not in memory["Scenes"]:
        memory["Scenes"].append(obs["current_scene"])
    if obs["interactable_items"]:
        if ("Interactable items in scene " + obs["current_scene"]) in memory["Interactable items in each scene"] and isinstance(memory["Interactable items in each scene"]["Interactable items in scene " + obs["current_scene"]], dict):
            for item in obs["interactable_items"]:
                if item in memory["Interactable items in each scene"]["Interactable items in scene " + obs["current_scene"]] and memory["Interactable items in each scene"]["Interactable items in scene " + obs["current_scene"]][item] == (obs["interactable_items"][item] + " Has been investigated."):
                    obs["interactable_items"][item] = obs["interactable_items"][item] + " Has been investigated."
        memory["Interactable items in each scene"]["Interactable items in scene " + obs["current_scene"]] = obs["interactable_items"]
    else:
        memory["Interactable items in each scene"]["Interactable items in scene " + obs["current_scene"]] = "Explored, no interactable items remaining"
    for adjacent_scene in obs["adjacent_scenes"]:
        if adjacent_scene not in memory["Scenes"]:
            memory["Scenes"].append(adjacent_scene)
        if ("Interactable items in scene " + adjacent_scene) not in memory["Interactable items in each scene"]:
            memory["Interactable items in each scene"]["Interactable items in scene " + adjacent_scene] = "Hasn't explore yet"
    if obs["tools_in_bag"]:
        memory["tools_in_bag"] = obs["tools_in_bag"]
    else:
        memory["tools_in_bag"] = "empty"
    memory["Current Location"] = obs["current_scene"]
    for adjacent_scene in obs["adjacent_scenes"]:
        if (obs["current_scene"], adjacent_scene) not in memory["Map of Scenes (edge list)"] and (adjacent_scene, obs["current_scene"]) not in memory["Map of Scenes (edge list)"]:
            memory["Map of Scenes (edge list)"].append((obs["current_scene"], adjacent_scene))

    if investigated is not None:
        tmp = memory["Interactable items in each scene"]["Interactable items in scene " + obs["current_scene"]][investigated].replace(" Has been investigated.", "")
        memory["Interactable items in each scene"]["Interactable items in scene " + obs["current_scene"]][investigated] = tmp + " Has been investigated."




class StateNode:
    def __init__(self, reward=0, done=False):
        self.state = None
        self.prev_state: StateNode = None
        self.prev_actions: str = None
        self.prev_action: str = None
        self.prev_response: str = None
        self.valid_actions = None
        self.action_probs = []

        self.N = 0
        self.children: List[ActionNode] = []
        self.reward = reward
        self.score = 0
        self.done = done

        self.use_subtask = False


class ActionNode:
    def __init__(self, action):
        self.action = action
        self.N = 0
        self.Q = 0
        self.Rs = []
        self.children: List[StateNode] = []
        self.children_text = []

client = OpenAI(api_key="")
TOTAL_SPENT = 0.0

def calculate_gpt4o_mini_cost(usage) -> float:
    """
    Calculates cost for gpt-4o-mini based on usage object.
    Pricing: $0.15/1M input tokens, $0.60/1M output tokens.
    """
    if not usage:
        return 0.0
        
    input_price_per_million = 0.15
    output_price_per_million = 0.60
    
    input_cost = (usage.prompt_tokens / 1_000_000) * input_price_per_million
    output_cost = (usage.completion_tokens / 1_000_000) * output_price_per_million
    
    return input_cost + output_cost

def calculate_o4_mini_cost(usage) -> float:
    """
    Calculates cost for o4-mini based on usage object.
    Pricing: $1.10/1M input tokens, $4.40/1M output tokens.
    """
    if not usage:
        return 0.0
        
    input_price_per_million = 1.10
    output_price_per_million = 4.40
    
    input_cost = (usage.prompt_tokens / 1_000_000) * input_price_per_million
    output_cost = (usage.completion_tokens / 1_000_000) * output_price_per_million
    
    return input_cost + output_cost

def calculate_o3_cost(usage) -> float:
    """
    Calculates cost for o3 based on usage object.
    Pricing: $2.00/1M input tokens, $8.00/1M output tokens.
    """
    if not usage:
        return 0.0
        
    input_price_per_million = 2.00
    output_price_per_million = 8.00
    
    input_cost = (usage.prompt_tokens / 1_000_000) * input_price_per_million
    output_cost = (usage.completion_tokens / 1_000_000) * output_price_per_million
    
    return input_cost + output_cost

def chat_completion_with_retries(model: str, sys_prompt: str, prompt: str, max_retries: int = 5, retry_interval_sec: int = 20, api_keys=None, **kwargs) -> Mapping:
    # Use the global variable
    global TOTAL_SPENT 

    for n_attempts_remaining in range(max_retries, 0, -1):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                **kwargs
            )

            # 2. Calculate and update cost upon success
            if model=="gpt-4o-mini":
                call_cost = calculate_gpt4o_mini_cost(res.usage)
            elif model=="o4-mini":
                call_cost = calculate_o4_mini_cost(res.usage)
            elif model=="o3":
                call_cost = calculate_o3_cost(res.usage)
            TOTAL_SPENT += call_cost

            print("sys_prompt: ", sys_prompt)
            print("user_prompt: ", prompt)
            print("-" * 20)
            print(f"Call Cost: ${call_cost:.6f}")
            print(f"Total Accumulated Cost: ${TOTAL_SPENT:.6f}")
            print("-" * 20)
            
            return res

        except (
            openai.RateLimitError,
            openai.APIError,
            openai.OpenAIError,
        ) as e:
            print(e)
            print(f"Hit openai.error exception. Waiting {retry_interval_sec} seconds for retry... ({n_attempts_remaining - 1} attempts remaining)", flush=True)
            time.sleep(retry_interval_sec)
            
    return {}

class LLMAgent:
    def __init__(self):
        self.model = "gpt-4o-mini"
        self.llm_temperature = 1
        self.softmax_temperature = 5

    def _format_state(self, state_node: StateNode):
        if state_node.prev_state is None:
            return f"""PREVIOUS ACTIONS for the SUBTASK: <s>
CURRENT_STATE: 
- Current Scene: {state_node.state["current_scene"]}
- Interactable Items (and its description) in Current Scene: {state_node.state["interactable_items"]}
- Adjacent Scenes: {state_node.state["adjacent_scenes"]}
- Tools in Bag: {state_node.state["tools_in_bag"]}"""
        else:
            return f"""PREVIOUS ACTIONS for the SUBTASK: {state_node.prev_actions}
CURRENT_STATE: 
- Current Scene: {state_node.state["current_scene"]}
- Interactable Items (and its description) in Current Scene: {state_node.state["interactable_items"]}
- Adjacent Scenes: {state_node.state["adjacent_scenes"]}
- Tools in Bag: {state_node.state["tools_in_bag"]}"""

    def get_probs_prompts(self, state_node: StateNode, proposed_subtask = None):
        formatted_state = self._format_state(state_node)
        actions_str = [f"{i}: {a}" for i, a in enumerate(state_node.valid_actions)]
        formatted_actions = "\n".join(actions_str)

        sys_prompt = """You are a player in a text-based adventure game. Your task is to evaluate and select next action based on subtask.
You can perform one of the following actions:
- click(<interactable item>): Click an <interactable item> to examine it or interact with it. For example, you can examine a door handle that is marked as interactable.
- apply(<applicable tool>, <interactable item>): Apply an <applicable tool> in your bag to an <interactable item>. For example, you can apply a key in your bag to an interactable locked door to open it.
- move(<interactable scene>): Move to a nearby <interactable item> to further explore. For example, you can move to the living room to explore more interactable items there.
- craft(<applicable tool>, <applicable tool>): Use one <applicable tool> in bag to another <applicable tool> in bag to craft something new. For example, you can use a battery in bag to a controller in bag to craft a new charged controller."""

        if proposed_subtask is not None:
            user_prompt = f"""Your current subtask is: {proposed_subtask}
You are now facing the following state in the game: {formatted_state}
Considering the subtask and previous actions, please select the select next action from the following list:
{formatted_actions}
Respond by providing the index of the action only. Your response should be a single integer, without any extra formatting, spaces, punctuation, or text."""
        else:
            user_prompt = f"""You are now facing the following state in the game: {formatted_state}
Considering the previous actions, please select the select next action from the following list:
{formatted_actions}
Respond by providing the index of the action only. Your response should be a single integer, without any extra formatting, spaces, punctuation, or text."""


        print(user_prompt)


        return sys_prompt, user_prompt   

    def get_action_probs(self, state_node: StateNode, proposed_subtask):
        if state_node.prev_state is not None and state_node.prev_state.use_subtask is True:
            index = np.argmax(state_node.prev_state.action_probs[0])
            if state_node.prev_action == state_node.prev_state.children[index].action:
                sys_prompt, user_prompt = self.get_probs_prompts(state_node, proposed_subtask)
                state_node.use_subtask = True
            else:
                sys_prompt, user_prompt = self.get_probs_prompts(state_node)
                state_node.use_subtask = False
        else:
            sys_prompt, user_prompt = self.get_probs_prompts(state_node, proposed_subtask)
            state_node.use_subtask = True


        valid_labels = [str(i) for i in range(len(state_node.valid_actions))]
        
        print("***sys_prompt***:", sys_prompt)
        print("***user_prompt***:", user_prompt)

        res = chat_completion_with_retries(
            model=self.model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=2,
            temperature=self.llm_temperature,
            logprobs=True,
            top_logprobs=min(len(state_node.valid_actions), 20)
        )

        text = res.choices[0].message.content

        top_logprobs = res.choices[0].logprobs.content[0].top_logprobs
        
        action_log_dict = {}
        for i, logprob in enumerate(top_logprobs):
            action_token = logprob.token
            action_logprob = logprob.logprob
            action_log_dict[action_token] = action_logprob

        logprobs_list = [action_log_dict.get(label, -5) for label in valid_labels]
        probs_list = softmax(logprobs_list, self.softmax_temperature)
        
        return text, probs_list
    
    def get_response(self, sys_prompt, user_prompt, model=None):
        if model:
            res = chat_completion_with_retries(
                model=model,
                sys_prompt=sys_prompt,
                prompt=user_prompt,
                temperature=self.llm_temperature,
            )
        else:
            res = chat_completion_with_retries(
                model=self.model,
                sys_prompt=sys_prompt,
                prompt=user_prompt,
                temperature=self.llm_temperature,
            )
        
        text = res.choices[0].message.content

        return text

def search(graph: Graph, log_f, proposed_subtask, llm: LLMAgent):
    obs, valid_actions = get_obs_and_valid_actions(graph)

    formatted_state = f"""CURRENT_STATE: 
- Current Scene: {obs["current_scene"]}
- Interactable Items (and its description) in Current Scene: {obs["interactable_items"]}
- Adjacent Scenes: {obs["adjacent_scenes"]}
- Tools in Bag: {obs["tools_in_bag"]}"""
    actions_str = [f"{i}: {a}" for i, a in enumerate(valid_actions)]
    formatted_actions = "\n".join(actions_str)

    sys_prompt = """You are a player in a text-based adventure game. Your task is to evaluate and select next action based on subtask.
You can perform one of the following actions:
- click(<interactable item>): Click an <interactable item> to examine it or interact with it. For example, you can examine a door handle that is marked as interactable.
- apply(<applicable tool>, <interactable item>): Apply an <applicable tool> in your bag to an <interactable item>. For example, you can apply a key in your bag to an interactable locked door to open it.
- move(<interactable scene>): Move to a nearby <interactable item> to further explore. For example, you can move to the living room to explore more interactable items there.
- craft(<applicable tool>, <applicable tool>): Use one <applicable tool> in bag to another <applicable tool> in bag to craft something new. For example, you can use a battery in bag to a controller in bag to craft a new charged controller."""

    user_prompt = f"""Your current subtask is: {proposed_subtask}
You are now facing the following state in the game: {formatted_state}
Considering the subtask and previous actions, please select the select next action from the following list:
{formatted_actions}
Respond by providing the index of the action only. Your response should be a single integer, without any extra formatting, spaces, punctuation, or text."""

    res = chat_completion_with_retries(
        model=llm.model,
        sys_prompt=sys_prompt,
        prompt=user_prompt,
        max_tokens=2,
        temperature=llm.llm_temperature
    )

    
    text = res.choices[0].message.content

    return valid_actions[int(text)]



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
def format_previous_actions_and_rewards(previous_actions_and_rewards):
    if not previous_actions_and_rewards:
        return "empty"
    
    n = len(previous_actions_and_rewards)
    formatted_items = []
    
    for i, (action, reward) in enumerate(previous_actions_and_rewards):
        time_tag = i - n
        formatted_items.append(f"t{time_tag}: {action}: {reward}")
    
    return f"[{'; '.join(formatted_items)}]"

if __name__ == "__main__":
    game_index = "3-1"
    game_path = os.path.join("..", "data", f"game{game_index}.yaml")
    keylog_path = os.path.join("..", "data", "reference", f"key_log_{game_index}.txt")
    graph = Graph(game_path)
    key_log = parse_key_log(keylog_path)
    env.global_vars.reset_global_vars()

    memory = {}
    memory["Scenes"] = []
    memory["Map of Scenes (edge list)"] = []
    memory["tools_in_bag"] = "empty"
    memory["Current Location"] = "None"
    memory["Interactable items in each scene"] = {}
    llm = LLMAgent()

    accumulated_reward = 0

    graph.describe()
    obs, _ = get_obs_and_valid_actions(graph)
    update_memory(memory, obs)
    previous_actions_and_rewards = deque(maxlen=10)

    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    with open(os.path.join("log", f"game{game_index}_ReAct_{formatted_time}"), "w") as log_f:
        for step in range(0, 100):
            print("Step:", step+1)
            print("Step:", step+1, file=log_f)

            if "input(" in key_log[0][1]:
                action = key_log[0][1]
                graph.current_scene = key_log[0][0]
            else:
                
                previous_actions_and_rewards_str = format_previous_actions_and_rewards(previous_actions_and_rewards)
                proposed_subtask_sys_prompt = "You are in a text-based adventure game"
                proposed_subtask_user_prompt = f"""Current State: {memory}
Your previous actions and reward: {previous_actions_and_rewards_str}
Considering the Current State, your previous actions and reward, propose the next subtask, and how to achieve the subtask, the next subtask should be small.
You can perform one of the following actions:
- click(<interactable item>): Click an <interactable item> to examine it or interact with it. For example, you can examine a door handle that is marked as interactable. You cannot click tools in bag.
- apply(<applicable tool>, <interactable item>): Apply an <applicable tool> in your bag to an <interactable item>. For example, you can apply a key in your bag to an interactable locked door to open it.
- move(<interactable scene>): Move to a nearby <interactable item> to further explore. For example, you can move to the living room to explore more interactable items there.
- craft(<applicable tool>, <applicable tool>): Use one <applicable tool> in bag to another <applicable tool> in bag to craft something new. For example, you can use a battery in bag to a controller in bag to craft a new charged controller.
**Note** If you decide to move to another scene, you have to plan the path from your current scene to destination scene based on Map of Scenes (edge list)."""
                proposed_subtask = llm.get_response(proposed_subtask_sys_prompt, proposed_subtask_user_prompt, "o3")
                print("proposed_subtask_user_prompt:", proposed_subtask_user_prompt, file=log_f)
                print("Proposed_subtask:", proposed_subtask, file=log_f)

                action = search(graph, log_f, proposed_subtask, llm)
                print(action)

            print("Action:", action)
            print("Action:", action, file=log_f)
            current_scene_str = graph.current_scene
            graph.describe()
            response = graph.react(action)
            print("Reward:", response)
            print("Reward:", response, file=log_f)

            if "input(" in key_log[0][1]:
                previous_actions_and_rewards.append((f"move({current_scene_str})", "Action executed successfully"))
            else: 
                previous_actions_and_rewards.append((action, response))

            graph.describe()
            obs, _ = get_obs_and_valid_actions(graph)
            if "click(" in action and "nothing happen" in response.lower():
                tmp = action.removeprefix("click(")
                tmp = tmp.removesuffix(")")
                update_memory(memory, obs, tmp)
            else:
                update_memory(memory, obs)
            print(f"Updated memory: {memory}", file=log_f)

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