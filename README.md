# Distributional Reinforcement Learning - Optimal Execution (Weston 2020)

## File Descriptions 
**Run_me.py** - This file includes global params for traning the model. I'll summarize the customizatios possible soon. The output of the file is sent to Wandb - a model tracker much like MLFlow. To set up, pip install wandb and link to your account. I'll add a graph of what the output should be like here - it tracks actions, states, rewards, etc.
* **market.k** - Temporary Impact - in the paper, the author considers just the case where price impact is only decided by some coefficient times the number of shares traded - (Almgren & Chriss 2000)
* **market.b** - Permanent Impact - in academia, there's supposedly a distinction between temporary and permanent impact (this figure again is assumed to be a constant real number)
* 

**market_modelsM.py** - This includes the price dynamics for the basic simulation. In addition to several price dynamics (in classes), there are a basic class of market models - not sure what these do in the grand scheme of things though/the paper they came from.


