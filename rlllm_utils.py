import numpy as np
import scipy.stats as stats
import random
from llm import generate

# --- Multi-Armed Bandit Environment ---
class MAB:
    def __init__(self, arm_probabilities):
        """
        Initialize the MAB.
        Args:
            arm_probabilities (list): A list of probabilities, where each probability corresponds to the chance of reward for an arm.
        """
        self.arm_probabilities = arm_probabilities
        self.n_arms = len(arm_probabilities)

    def pull_arm(self, arm_index):
        """
        Pull an arm and get a reward (0 or 1).
        Args:
            arm_index (int): The index of the arm to pull.
        Returns:
            int: 1 if reward, 0 otherwise.
        """
        if arm_index < 0 or arm_index >= self.n_arms:
            raise ValueError("Invalid arm index")
        return 1 if random.random() < self.arm_probabilities[arm_index] else 0

    def description(self):
        return f"""
            {self.n_arms}-armed bandit game. 
            Your goal is to choose the arm that you believe will give you the highest reward.

            A reward of 0 means no success, a reward of 1 means success.
        """

# --- Thompson Sampler Agent ---
class ThompsonSampler:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        # Beta distribution parameters (alpha, beta) for each arm
        # Initializing with (1, 1) which corresponds to a uniform prior
        self.beta_params = np.ones((n_arms, 2))

    def select_arm(self):
        """
        Select an arm based on Thompson sampling.
        """
        sampled_theta = [stats.beta.rvs(a=self.beta_params[i, 0], b=self.beta_params[i, 1])
                         for i in range(self.n_arms)]
        return np.argmax(sampled_theta)

    def update(self, arm_index, reward):
        """
        Update the Beta distribution for the pulled arm.
        Args:
            arm_index (int): The arm that was pulled.
            reward (int): The reward received (1 or 0).
        """
        if reward == 1:
            self.beta_params[arm_index, 0] += 1  # Increment alpha (successes)
        else:
            self.beta_params[arm_index, 1] += 1  # Increment beta (failures)

    def get_posterior_means(self):
        return [self.beta_params[i, 0] / (self.beta_params[i, 0] + self.beta_params[i, 1])
                for i in range(self.n_arms)]

# --- LLM Agent ---
class LLMAgent:
    def __init__(self, mab, generation_instructions, temperature=0.7):
        """
        Initialize the LLM agent.
        Args:
            n_arms (int): Number of arms in the MAB.
            temperature (float): Temperature for LLM generation.
        """
        self.mab = mab
        self.temperature = temperature
        self.generation_instructions = generation_instructions
        self.history = [] # List of (arm_index, reward) tuples

    def select_arm(self):
        """
        Select an arm using the LLM.
        Args:
            observations_O (list): A list of (arm_index, reward) tuples.
        Returns:
            int: The arm index (0-based) chosen by the LLM.
                  Returns -1 if LLM output is invalid.
        """
        prompt = format_prompt(self.mab.description, self.history, self.generation_instructions)
        chosen_arm_text = generate(prompt, self.temperature)
        chosen_arm = int(chosen_arm_text.strip())  # do we need more error checking here?

        if 0 <= chosen_arm < self.n_arms:
            return chosen_arm
        else:
            print(f"LLM Warning: Invalid arm choice '{chosen_arm_text}'. Defaulting to random.")
            return random.choice(range(self.n_arms)) # Fallback

    def update(self, arm_index, reward):
        self.history.append((arm_index, reward))

    def reset(self):
        self.history = []

def format_prompt(task_description, history, generation_instructions):
    """
    Formats the history of observations into a prompt for the LLM.
    Args:
        observations_O (list): A list of (arm_index, reward) tuples.
    """
    
    return task_description + format_history(history) + generation_instructions

def format_history(history):
    prompt = "Here is the history of your previous plays (arm chosen, reward received):\n"

    if len(history) == 0:
        prompt += "No plays yet.\n"
    else:
        for i, (arm, reward) in enumerate(history):
            prompt += f"Play {i}: Chose Arm {arm}, Received Reward {reward}\n"

    return history

