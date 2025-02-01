# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import random


def extract_solution(solution_str):
    pattern = r'<reward>(.*?)</reward>'
    # Find all matches to count total occurrences
    all_matches = re.findall(pattern, solution_str, re.DOTALL)
    count = len(all_matches)
    
    # Get first match content
    first_match = re.search(pattern, solution_str, re.DOTALL)
    content = first_match.group(1).strip() if first_match else None
    
    return content, count


def compute_score(solution_str: str, ground_truth: bool, score=1.):
    """The scoring function for Generative ORM.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    do_print = random.randint(1, 32) == 1
    SEP = 'Please first analyze and provide your reasoning for the action sequence and strategy, then provide the reward value (integer between 0 and 100) enclosed in <reward>...</reward> tags.'
    solution_str = solution_str.split(SEP)[-1]
    reward, count = extract_solution(solution_str=solution_str)

    BASE_SCORE = -1  # default score
    FORMAT_SCORE = 0.1

    if do_print:
        print(f"===================================")
        print(f"Ground truth resolved: {ground_truth}")
        print(f"Extracted reward: {reward}")
        print(f"------------BEGIN SOLUTION---------------")
        print(f"{solution_str}")
        print(f"------------END SOLUTION---------------")

    if reward is None:
        if do_print:
            print(f"No reward found in solution_str. Reward: {BASE_SCORE} (base score, incorrect format)")
        return BASE_SCORE
    
    if count > 1:
        if do_print:
            print(f"Multiple rewards found in solution_str. Reward: {BASE_SCORE} (base score, incorrect format)")
        return BASE_SCORE

    # Try convert reward to int
    try:
        reward = int(reward)
    except ValueError as e:
        if do_print:
            print(f"Failed to convert reward to int: {e}. Reward: {BASE_SCORE} (base score, incorrect format)")
        return BASE_SCORE # format score

    # Try assert reward is in [0, 100]
    if reward < 0 or reward > 100:
        if do_print:
            print(f"Reward is out of range: {reward}. Reward: {BASE_SCORE + FORMAT_SCORE} (format score, incorrect range)")
        return BASE_SCORE + FORMAT_SCORE # format score

    # Give reward score based on its closeness to ground truth
    target_reward = 100 if ground_truth else 0
    reward_diff = abs(reward - target_reward) # [0, 100]
    # scale diff to [-1, 1], when diff > 50, it is negative
    score = (50 - reward_diff) * 2 / 100
    # diff = 50, score = 0
    # diff = 0, score = 1
    # diff = 100, score = -1
    score += FORMAT_SCORE  # add format score

    if do_print:
        print(f"Target reward: {target_reward}. Difference: {reward - target_reward}. Reward score (+format score 0.1): {score}")
    return score
