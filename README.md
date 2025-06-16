We can use a LLM to play bandit problems.
Give the LLM the history of actions and rewards and ask it to produce the next action.

Hypothesis: the LLM will struggle to explore properly. LLMs don't sample from a posterior over optimal actions, they sample likely next tokens. I'm guessing these are very different things and seek to measure this difference.

Follow up.

What is the effect of prompting the LLM agent with RL theory? (similar in approach to ref 2.) Does this allow efficient exploration?

Related

1. LLMs Are In-Context Bandit Reinforcement Learners https://arxiv.org/pdf/2410.05362
2. Flipping Against All Odds: Reducing LLM Coin Flip
Bias via Verbalized Rejection Sampling https://arxiv.org/pdf/2506.09998