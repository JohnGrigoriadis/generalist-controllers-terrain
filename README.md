# Evolving Generalist Controllers for Varying Terrain Layouts
This repository contains the implementation for the research paper titled "Evolving Generalist Controllers for Varying Terrain Layouts". The paper addresses the limited understanding of robustness and generalisability in neuro-evolutionary methods, specifically focusing on artificial neural networks (ANNs) used in control tasks, such as those applied in robotics.

# Running Experiments and Simulations
1. The project uses the "BipedalWalker-v3" environment of the Gymnasium library, a continuation of the Gym library created for __Reinforcement Learning__ by OpenAI.
   To use the library you will need to run:

  `pip install gymnasium swig gymnasium[box2d]`

  Also, the other libraries used in the experiments can be installed using:

  `pip install numpy evotorch pytorch pandas time`

2. Run the experiments using the following command:
  `python XNES_BipedalWalker.py`

3. Run the visualisation of the results using the following command:
   `python visualiise_biped.py`
