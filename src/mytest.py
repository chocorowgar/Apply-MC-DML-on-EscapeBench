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
            call_cost = calculate_gpt4o_mini_cost(res.usage)
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

def LLM_inference(model, sys_prompt, user_prompt, llm_temperature):
    res = chat_completion_with_retries(
        model=model,
        sys_prompt=sys_prompt,
        prompt=user_prompt,
        temperature=llm_temperature,
    )
    
    text = res.choices[0].message.content

    return text

actions = [
    "click(telephone cabinet)",
    "move(To the telephone cabinet close-up)",
    "click(hairpin)",
    "click(telephone)",
    "apply(hairpin, drawer)",
    "click(carving knife)",
    "click(telephone)",
    "move(Back to the living room)",
    "move(To the bedroom)",
    "move(To the cardboard box close-up)",
    "apply(carving knife, cardboard box)",
    "click(door handle)",
    "move(Back to the bedroom)",
    "apply(door handle, bathroom door)",
    "click(hammer handle)",
    "click(bathroom door)",
]

def get_obs(graph: Graph):
    graph.describe()  # 需要 describe() 來更新可用的 action
    interactable_items = {}
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
            adjacent_scenes.append(name)
    for name, tool in env.global_vars.bag.tools.items():
        tools_in_bag[name] = tool.describe()
    obs = {"current_scene": graph.current_scene, "interactable_items": interactable_items, "adjacent_scenes": adjacent_scenes, "tools_in_bag": tools_in_bag}
    return obs

memory = {}
memory["Scenes"] = []
memory["Map of Scenes (edge list)"] = []
memory["tools_in_bag"] = "empty"
memory["Current Location"] = "None"
memory["Interactable items in each scene"] = {}

def update_memory(obs):
    if obs["current_scene"] not in memory["Scenes"]:
        memory["Scenes"].append(obs["current_scene"])
    if obs["interactable_items"]:
        memory["Interactable items in each scene"]["Interactable items in scene " + obs["current_scene"]] = obs["interactable_items"]
    else:
        memory["Interactable items in each scene"]["Interactable items in scene " + obs["current_scene"]] = "Explored, no interactable items remaining"
    for adjacent_scene in obs["adjacent_scenes"]:
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

if __name__ == "__main__":

    game_index = "1-1"
    game_path = os.path.join("..", "data", f"game{game_index}.yaml")
    keylog_path = os.path.join("..", "data", "reference", f"key_log_{game_index}.txt")
    graph = Graph(game_path)
    env.global_vars.reset_global_vars()

    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    with open(os.path.join("log", f"game{game_index}_mytest_{formatted_time}"), "w") as f:
        step = 0
        obs = get_obs(graph)
        update_memory(obs)
        print(f"Step: {step}", file=f)
        print(obs, file=f)
        print(memory, file=f)
        for action in actions:
            print("", file=f)
            step += 1
            print(f"Step: {step}", file=f)
            print(f"Action: {action}", file=f)
            response = graph.react(action)
            print(f"Response: {response}", file=f)
            obs = get_obs(graph)
            update_memory(obs)
            print(obs, file=f)
            print(memory, file=f)

    sys_prompt = "You are in a text-based adventure game"
    user_prompt = """Memory:{'Current Scene': 'living room', 'Interactable items in scene living room': {'telephone cabinet': 'An old-fashioned telephone cabinet stands here.', 'left door to the study': 'A locked door that leads to the study.', 'right door to the exit': 'A locked door that leads to the exit, the lock needs a 9-digit code all in numbers'}, 'tools_in_bag': 'empty', 'Interactable items in scene dining room and kitchen': "Hasn't explore yet", 'Interactable items in scene bedroom': "Hasn't explore yet"}
Considering the memory, propose the next subtask, and how to achieve the subtask, should be concise, the next subtask should be small
"""
    response = LLM_inference("gpt-4o-mini", sys_prompt, user_prompt, 0)

    print(response)

