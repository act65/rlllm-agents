import pytest
import numpy as np
from llm_explorer.rlllm_utils import MAB, ThompsonSampler, LLMAgent, format_prompt, format_history

# Fixtures
@pytest.fixture
def mab_env():
    return MAB(arm_probabilities=[0.1, 0.9, 0.3])

@pytest.fixture
def thompson_sampler(mab_env):
    return ThompsonSampler(n_arms=mab_env.n_arms)

@pytest.fixture
def llm_agent_env(mab_env):
    # Mock the MAB for LLMAgent since it's passed to constructor
    return LLMAgent(mab=mab_env, generation_instructions="Choose an arm.")

# Tests for MAB
def test_mab_initialization(mab_env):
    assert mab_env.n_arms == 3
    assert mab_env.arm_probabilities == [0.1, 0.9, 0.3]

def test_mab_pull_arm(mab_env):
    # Test pulling a valid arm
    reward = mab_env.pull_arm(1)
    assert reward in [0, 1]
    # Test pulling an invalid arm
    with pytest.raises(ValueError):
        mab_env.pull_arm(5)

def test_mab_description(mab_env):
    desc = mab_env.description()
    assert "3-armed bandit game" in desc
    assert "highest reward" in desc

# Tests for ThompsonSampler
def test_thompson_sampler_initialization(thompson_sampler, mab_env):
    assert thompson_sampler.n_arms == mab_env.n_arms
    assert thompson_sampler.beta_params.shape == (mab_env.n_arms, 2)
    assert np.all(thompson_sampler.beta_params == 1) # Default alpha=1, beta=1

def test_thompson_sampler_select_arm(thompson_sampler, mab_env):
    arm = thompson_sampler.select_arm()
    assert 0 <= arm < mab_env.n_arms

def test_thompson_sampler_update(thompson_sampler):
    arm_index = 0
    # Test update with reward = 1
    initial_alpha = thompson_sampler.beta_params[arm_index, 0]
    initial_beta = thompson_sampler.beta_params[arm_index, 1]
    thompson_sampler.update(arm_index, 1)
    assert thompson_sampler.beta_params[arm_index, 0] == initial_alpha + 1
    assert thompson_sampler.beta_params[arm_index, 1] == initial_beta

    # Test update with reward = 0
    initial_alpha = thompson_sampler.beta_params[arm_index, 0]
    initial_beta = thompson_sampler.beta_params[arm_index, 1]
    thompson_sampler.update(arm_index, 0)
    assert thompson_sampler.beta_params[arm_index, 0] == initial_alpha
    assert thompson_sampler.beta_params[arm_index, 1] == initial_beta + 1

def test_thompson_sampler_get_posterior_means(thompson_sampler):
    # After some updates
    thompson_sampler.update(0, 1)
    thompson_sampler.update(0, 1)
    thompson_sampler.update(1, 0)
    thompson_sampler.update(2, 1)
    means = thompson_sampler.get_posterior_means()
    assert len(means) == thompson_sampler.n_arms
    # Expected: arm0: (1+2)/(1+2+1) = 3/4 = 0.75
    # arm1: 1/(1+1+1) = 1/3 ~ 0.333
    # arm2: (1+1)/(1+1+1) = 2/3 ~ 0.666
    assert means[0] == (1+2) / (1+2+1) # alpha / (alpha + beta)
    assert means[1] == 1 / (1+1+1)
    assert means[2] == (1+1) / (1+1+1)


# Tests for LLMAgent
# Skipping select_arm for now as it involves actual LLM calls
# and requires mocking the 'generate' function.

def test_llm_agent_initialization(llm_agent_env, mab_env):
    assert llm_agent_env.mab == mab_env
    assert llm_agent_env.temperature == 0.7 # default
    assert llm_agent_env.generation_instructions == "Choose an arm."
    assert llm_agent_env.history == []

def test_llm_agent_update(llm_agent_env):
    llm_agent_env.update(0, 1)
    assert llm_agent_env.history == [(0, 1)]
    llm_agent_env.update(1, 0)
    assert llm_agent_env.history == [(0, 1), (1, 0)]

def test_llm_agent_reset(llm_agent_env):
    llm_agent_env.update(0, 1)
    llm_agent_env.reset()
    assert llm_agent_env.history == []

# Tests for helper functions
def test_format_history_empty():
    prompt = format_history([])
    assert "No plays yet." in prompt

def test_format_history_with_plays():
    history = [(0, 1), (2, 0)]
    prompt = format_history(history)
    assert "Play 0: Chose Arm 0, Received Reward 1" in prompt
    assert "Play 1: Chose Arm 2, Received Reward 0" in prompt

def test_format_prompt():
    task_desc = "This is a task."
    history = [(0,1)]
    gen_instr = "Do this."
    full_prompt = format_prompt(task_desc, history, gen_instr)
    # The format_history output is tested separately.
    # Here, just check if all parts are included.
    assert task_desc in full_prompt
    # format_history(history) will be part of it
    assert "Play 0: Chose Arm 0, Received Reward 1" in full_prompt
    assert gen_instr in full_prompt
