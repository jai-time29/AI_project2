# AI_project2
VisionScout — RL-Powered Smart Inspection : System that uses computer vision to detect product defects efficiently. An agent learns an optimal inspection policy using Q-learning, selecting suspicious image regions based on rewards and inspection costs.





## Model Evaluation

### AUROC Comparison
![AUROC Comparison](figures/01_auroc_comparison.png)

### Confusion Matrix
![Confusion Matrix](figures/02_confusion_matrix.png)

### Defect Rate by Suspicion Level
![Suspicion Analysis](figures/03_defect_rate_by_suspicion.png)

### Inspection Efficiency
![Inspection Efficiency](figures/04_inspection_efficiency.png)

### Q-learning Training Reward
![Training Reward](figures/06_training_reward.png)





## 🔍 Defect Visualization & Anomaly Localization

VisionScout divides each magnetic tile image into a **7 × 7 grid** and calculates an anomaly score for each region. The visualizations below show how the system identifies potentially defective areas.

### Good vs Defective Tiles

This comparison presents original images, anomaly-score overlays, and suspicious regions alongside ground-truth defect outlines.

![VisionScout comparison grid](figures/comparison_grid.png)

### Individual Defect Examples

**1. Blowhole Detection**

![Blowhole anomaly visualization](figures/defect_MT_Blowhole_exp2_num_308043.png)

**2. Crack Detection**

![Crack anomaly visualization](figures/defect_MT_Crack_exp1_num_339819.png)

**3. Break Detection**

![Break anomaly visualization](figures/defect_MT_Break_exp5_num_105879.png)

**4. Uneven Surface Detection**

![Uneven surface anomaly visualization](figures/defect_MT_Uneven_exp2_num_274120.png)

### Understanding the Visualizations

Each individual visualization contains four panels:

- **Original:** The input magnetic tile image.
- **Anomaly Heatmap:** Regional anomaly scores on the 7 × 7 grid.
- **Score Overlay:** Anomaly scores overlaid on the original image.
- **Suspicious Regions:** Orange boxes indicate medium-suspicion regions, red boxes indicate high-suspicion regions, and cyan outlines represent ground-truth defects when available.

These boxes highlight suspicious grid regions; they do not represent precise pixel-level defect segmentation.

### Why These Visualizations Matter

- **Interpretability:** Makes regional anomaly scores easier to understand.
- **Defect localization:** Highlights areas that may require closer inspection.
- **Ground-truth comparison:** Helps evaluate whether suspicious regions overlap actual defects.
- **Inspection efficiency:** Supports prioritizing suspicious regions instead of inspecting all regions equally.






## 📊 Experimental Results

VisionScout combines Gaussian-based anomaly detection with a Q-learning inspection agent to prioritize suspicious regions of magnetic tile images.

### Test Dataset

| Dataset property | Value |
|---|---:|
| Total test images | 266 |
| Good images | 190 |
| Defective images | 76 |
| Regions per image | 49 (7 × 7 grid) |

### Image-Level Classification Performance

| Metric | Result |
|---|---:|
| Accuracy | **96.24%** |
| Defect precision | **100.00%** |
| Defect recall | **86.84%** |
| Specificity | **100.00%** |
| False-reject rate on good tiles | **0.00%** |
| Missed-defect rate | **13.16%** |

### Confusion Matrix

| Actual / Predicted | Good | Defective |
|---|---:|---:|
| Good | 190 | 0 |
| Defective | 10 | 66 |

The agent correctly classified 190 good tiles and detected 66 of 76 defective tiles. Ten defective tiles were missed.

### Inspection Efficiency

| Metric | Result |
|---|---:|
| Mean regions inspected per tile | 4.36 |
| Median regions inspected per tile | 1 |
| Total regions in a full inspection | 49 |
| Estimated reduction in inspections | **91.11%** |

The estimated reduction compares the average number of regions inspected by the agent with inspecting all 49 regions in every tile.

### Reinforcement Learning Configuration

| Parameter | Value |
|---|---|
| Algorithm | Tabular Q-learning |
| Training episodes | 20,000 |
| Learning rate (α) | 0.10 |
| Discount factor (γ) | 0.95 |
| Initial exploration rate (ε) | 1.00 |
| Recorded final exploration rate | Approximately 0.067 |
| Maximum inspections per tile | 20 |
| State space | 135 possible states |
| Action space | 5 actions |

### Interpretation

- **High precision:** In the recorded evaluation, every tile predicted defective was actually defective.
- **High good-tile acceptance:** No good tiles were falsely rejected in this evaluation.
- **Defect detection:** The agent detected 86.84% of defective test tiles.
- **Inspection efficiency:** The recorded policy inspected an average of approximately 4.36 regions per tile instead of all 49.

### Limitations

The current evaluation missed 10 of the 76 defective test tiles. A zero false-reject rate on this test set does not guarantee zero false rejects on future data.

The reported results should be interpreted in the context of the current dataset split and evaluation procedure. The test set should remain independent of detector fitting, threshold selection, and hyperparameter tuning for a reliable final performance estimate.

The inspection-reduction figure measures the number of regions inspected, not elapsed time or computational cost.

### Detailed Results and Reproducibility

For the complete experimental results, saved evaluation metrics, training information, and configuration details, see **[RESULTS.md](RESULTS.md)**.

The machine-readable summary is available in [`results_summary.json`](results_summary.json).




















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
