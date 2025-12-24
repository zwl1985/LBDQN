# LBDQN
Implementation of "Deep Reinforcement Learning-Based Algorithm for Maximizing the Influence of Location-Based Social Networks"

## Project Structure

```
.
├── pre-data/
│   ├── process_foursquare.ipynb     # Data preprocessing notebook for Foursquare dataset
│   └── process_foursquare.py        # Data preprocessing script for Foursquare dataset
├── agent.py                         # Agent implementation with Q-network
├── environment.py                   # Graph environment for reinforcement learning
├── model.py                         # Neural network model definitions
├── main.py                          # Main training script
├── test.py                          # Testing and evaluation script
├── runner.py                        # Training runner
├── rl_utils.py                      # Reinforcement learning utilities
├── scalability_analysis.py          # Scalability analysis tools
├── compute_foursquare_ps.py         # Calculate comprehensive relevance 
├── compute_foursquare_new_ps.py     # Calculate comprehensive relevance 
└── README.md
```

## Key Components

### 1. Core Modules

- **agent.py**: Implements the reinforcement learning agent with Deep Q-Network architecture
- **environment.py**: Defines the graph environment where the influence maximization problem is solved
- **model.py**: Contains the neural network models used for node representation learning
- **rl_utils.py**: Utility functions for reinforcement learning operations

### 2. Main Scripts

- **main.py**: Entry point for training the LBDQN model
- **test.py**: Evaluation script for testing the trained model on influence maximization tasks
- **runner.py**: Orchestrates the training process
- **scalability_analysis.py**: Analyzes how the algorithm scales with network size

### 3. Data Processing

- **pre-data/process_foursquare.ipynb**: Jupyter notebook for preprocessing the Foursquare location-based social network dataset
- **pre-data/process_foursquare.py**: Script for preprocessing the Foursquare location-based social network dataset
- **compute_foursquare_ps.py** & **compute_foursquare_new_ps.py**: Scripts for computing comprehensive relevance

## Usage

### Download dataset
- [brightkite](https://snap.stanford.edu/data/loc-Gowalla.html)
- [Gowalla](https://snap.stanford.edu/data/loc-Gowalla.html)
- [Foursquare](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)

### Preprocess
For foursquare dataset, run:
```bash
python pre-data/process_foursquare.py # init processing
```
```bash
python compute_foursquare_ps.py # compute comprehensive relevance
```
Other datasets can be processed by making the appropriate modifications to the scripts above.

### Training

To train the model, run:
```bash
python main.py
```

### Testing

To evaluate the trained model:
```bash
python test.py --graph_name brightkite --location_counts 1 --folder_name pth/lbdqn.model
```

### Arguments for test.py

- `-gn`, `--graph_name`: Graph name (default: 'brightkite')
- `-ad`, `--aggr_direction`: Aggregation direction (default: 'all')
- `-fn`, `--folder_name`: Model parameter path (default: 'pth/lbdqn.model')
- `-lc`, `--location_counts`: Number of locations (1, 5, 10, 15, 20) (default: 1)
- `-p`, `--processes`: Number of processes for parallel execution (default: 4)

## Requirements

- Python 3.10.6
- torch 2.1.0+cu121
- torch-geometric 2.7.0
- torch-scatter 2.1.2+pt21cu121
- networkx 2.8.8
- numpy 1.22.2
- pandas 1.5.2
- matplotlib 3.7.2
- seaborn 0.13.2
- scipy 1.11.0
- tqdm 4.65.0

## License

[LICENSE](LICENSE)
