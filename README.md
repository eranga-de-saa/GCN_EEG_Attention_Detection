# COSC-521 Project: Graph Convolutional Neural Networks for EEG-based Stroop Task Classification

This project implements a Graph Convolutional Neural Network (GCNN) to classify EEG signals collected during Stroop color-naming tasks. The network analyzes brain connectivity patterns to distinguish between different experimental conditions using data collected from MUSE wearable EEG devices.

## Dataset and Experiment Design

### Stroop Color-Naming Task

The Stroop task requires participants to identify the color of a word displayed on screen. Two types of trials are used:

- **Congruent trials**: The word's color matches its meaning (e.g., the word "RED" displayed in red)
- **Incongruent trials**: The word's color does not match its meaning (e.g., the word "RED" displayed in blue)

### Experimental Setup

**Participants**: 20 healthy adults (age range: 20-31 years)

**Task Parameters**:
- 4 color options: red, green, blue, yellow
- Color-coded keyboard for responses
- 4 blocks of 48 trials each (192 total trials per participant)
- Inter-stimulus interval: 1.5 seconds with fixation cross

**Conditions**:
- **Distractor (D) trials**: Sound interference (ring tones) during Stroop presentation
- **Non-distractor (ND) trials**: No sound interference
- 2 blocks with sound interference, 2 blocks without

### EEG Data Collection

**Device**: MUSE wearable EEG headband
- 4 electrodes: TP9, AF7, AF8, TP10
- Sampling rate: 256 Hz
- Positioned according to 10-20 international system

## Data Preprocessing Pipeline

1. **Band-pass filtering**: 0.5-45 Hz to remove unwanted frequencies
2. **Artifact removal**: Elimination of muscle artifacts (eye blinks, jaw clenches)
3. **Segmentation**: 1.5-second trials based on Stroop word onset
4. **Adjacency matrix construction**: Pearson correlation coefficients between electrodes
5. **Identity matrix subtraction**: Remove self-correlations

The preprocessing generates adjacency matrices representing functional connectivity between brain regions for each trial.

## Graph Convolutional Neural Network (GCNN)

### Architecture

The GCNN model leverages the graph structure of EEG electrode connectivity:

- **Input**: Adjacency matrices representing electrode correlations
- **Node features**: Learnable embeddings (no predefined features)
- **Message passing**: Aggregation of neighboring electrode information
- **Output**: Binary classification (congruent vs. incongruent, distractor vs. non-distractor)

### Message Passing Formula

```
h^(k+1)_u = UPDATE^(k)(h^(k)_u, AGGREGATE^(k)(h^(k)_v, ∀v ∈ N(u)))
```

Where:
- `h^(k)_u`: Node u's embedding at layer k
- `N(u)`: Neighborhood of node u
- UPDATE and AGGREGATE are learnable functions

### Model Configurations

Four network configurations were tested:

1. **3L_32h**: 3 hidden layers, 32-dimensional embeddings
2. **3L_16h**: 3 hidden layers, 16-dimensional embeddings  
3. **2L_32h**: 2 hidden layers, 32-dimensional embeddings
4. **2L_16h**: 2 hidden layers, 16-dimensional embeddings

## Training Details

- **Framework**: PyTorch Geometric
- **Data split**: 80% training / 20% testing (participant-wise split)
- **Epochs**: 300
- **Objective**: Cross-participant generalization


## Key Features

- **Graph-based approach**: Treats EEG electrodes as nodes in a connectivity graph
- **Functional connectivity**: Uses Pearson correlation for edge weights
- **Cross-participant validation**: Tests generalization to unseen participants
- **Multi-condition classification**: Handles both congruency and distractor effects
- **Learnable node embeddings**: Captures electrode-specific characteristics

## Results

The model was evaluated on two main classification tasks:

1. **Incongruent trials**: Distractor vs. Non-distractor classification
2. **Congruent trials**: Distractor vs. Non-distractor classification

Training and testing accuracies were compared across all four model configurations to identify optimal architectures for EEG-based Stroop task classification.

## Dependencies

- PyTorch
- PyTorch Geometric
- NumPy
- SciPy
- Pandas
- MNE-Python (for EEG preprocessing)
- Matplotlib/Seaborn (for visualization)

## Usage

1. **Model training**: Execute GCNN training scripts with desired configuration
2. **Evaluation**: Analyze cross-participant classification performance

This project demonstrates the effectiveness of graph neural networks for analyzing brain connectivity patterns in cognitive tasks, providing insights into neural mechanisms underlying attention and cognitive control.
