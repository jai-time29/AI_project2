# AI_project2
VisionScout — RL-Powered Smart Inspection : System that uses computer vision to detect product defects efficiently. An agent learns an optimal inspection policy using Q-learning, selecting suspicious image regions based on rewards and inspection costs.

















































































## How to Use It

VisionScout combines visual anomaly detection with a Reinforcement Learning (RL) agent to make inspection decisions efficiently. The system assigns anomaly scores to image regions, categorizes their suspicion levels, and uses a trained agent to decide which regions to inspect and whether to accept or reject a tile.

### 1. Clone the Repository

```bash
git clone https://github.com/jai-time29/AI_project2.git
cd AI_project2
```

### 2. Set Up the Environment

Create and activate a virtual environment.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### 3. Prepare the Dataset

Place the dataset in the directory structure expected by the data-loading and feature-extraction scripts. Ensure the dataset paths and configuration match those specified in the project scripts.

The dataset should contain the image tiles and the corresponding labels needed for training and evaluation.

> **Note:** The exact dataset directory structure depends on the configuration used by the data-loading scripts.

### 4. Generate Anomaly Scores

Run the anomaly-scoring script:

```bash
python compute_scores.py
```

This step calculates anomaly scores for image regions using a Gaussian-based anomaly detection method and saves the outputs required by the inspection environment.

The generated `scores.pt` file contains the anomaly-scoring results and related calibration information used by subsequent steps.

### 5. Create the Inspection Environment

Run:

```bash
python step5_environment.py
```

This step prepares the reinforcement-learning inspection environment using the anomaly scores. The environment defines the available actions, inspection costs, rewards, and stopping decisions.

The agent can:
- Inspect high-suspicion regions.
- Inspect medium-suspicion regions.
- Inspect low-suspicion regions.
- Accept a tile.
- Reject a tile.

### 6. Train the Q-Learning Agent

Run:

```bash
python RL_q_learning.py
```

The Q-learning agent learns an inspection policy by interacting with the environment. It learns which regions to inspect and when to accept or reject a tile based on the reward structure.

The training process produces the learned Q-table and evaluation outputs, including:

- `step6_q_table.pt` — the trained Q-learning policy.
- `step6_test_metrics.json` — evaluation metrics on the test split.
- `step6_training_log.csv`
