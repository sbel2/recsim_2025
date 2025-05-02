📚 RecSimRL: A Lightweight Simulator for Long-Term Recommendation
This repository contains a lightweight simulation environment and agent implementations for studying long-term user engagement in personalized recommendation systems. Built on Google’s RECSIM, the simulator supports dynamic user interest modeling, content generation, and evaluation of both simple and deep reinforcement learning agents.

🚀 Support for various agents:

- Random Agent
- Best Policy (Oracle)
- UCB Bandit
- Contextual Bandits (DocOnly, UserDoc)
- Naive DQN
- Double DQN

Evironment code files (can be readapted to other simulator projects): 
- interest_evolution.py     (Core environment definition)
- runner.py                 (Training and evaluation loop)
- helper file                   (Environment utilities, choice model, etc.)

To run code, write in terminal:
python main.py
Results (e.g., reward curves) are saved to ./logs/ and can be visualized with the provided plotting scripts in Visualization/.

📊 Evaluation Metrics
Total Watch Time: Cumulative engagement across an episode
Recovered Time: Additional time added back to user budget due to rewarding content
CTR (optional): If implemented in logging

📖 Citation
Built upon concepts from:
RecSim (Ie et al., 2019)
