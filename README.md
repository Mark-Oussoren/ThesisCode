
# Distributional Reinforcement Learning - Optimal Execution (Weston 2020)

## File Descriptions 
**Run_me.py** - This file includes global params for training the model. I'll summarize the params below (WIP). The output of the file is sent to Wandb - a model tracker much like MLFlow. To set up, pip install wandb and link to your account. I'll add a graph of what the output should be like here - it tracks actions, states, rewards, etc.
* **market.k** - Temporary Impact - in the paper, the author considers just the case where price impact is only decided by some coefficient times the number of shares traded - (Almgren & Chriss 2000)
* **market.b** - Permanent Impact - in academia, there's supposedly a distinction between temporary and permanent impact (this figure again is assumed to be a constant real number)
* 

**market_modelsM.py** - This includes the price dynamics for the basic simulation. In addition to several price dynamics (in classes), there are a basic class of market models - not sure what these do in the grand scheme of things though/the paper they came from.
Classes
* bs_stock
  * Params: S0 (initial price), drift, vol (daily volatility), n_steps (times agent can trade)
  * Functions: reset, generate_price 
    * reset restarts the episode to the initial price 
    * Price evolution in generate_price: <a href="https://www.codecogs.com/eqnedit.php?latex=S_t=S_{t-1}e^{(\mu&space;-0.5\sigma^2)\Delta&space;t&plus;\sigma\sqrt{\Delta&space;t}Z_t)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?S_t=S_{t-1}e^{(\mu&space;-0.5\sigma^2)\Delta&space;t&plus;\sigma\sqrt{\Delta&space;t}Z_t)}" title="S_t=S_{t-1}e^{(\mu -0.5\sigma^2)\Delta t+\sigma\sqrt{\Delta t}Z_t)}" /></a> 
* mean_rev_stock
* signal_stock
* real_stock
* real_stock_lob
* market
* lob_market



