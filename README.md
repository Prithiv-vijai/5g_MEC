# Project Title

## Overview

5G resource allocation optimization using advanced machine learning algorithms

## Table of Contents

1. [Installation](#installation)
2. [Steps to Run](#steps-to-run)


## Installation

To get started, clone this repository and navigate to the project directory.

```bash
git clone https://github.com/Prithiv-vijai/5g_MEC.git
cd 5g_MEC
```

### Optional Step: Set Up a Virtual Environment

It's recommended to create a virtual environment for this project to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate
cd src  # On Windows use `venv\Scripts\activate`
```

### Install Requirements

Install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Steps to Run

Follow these steps to run the project:

1. **Preprocess the Dataset**
   - Run `preprocess.py`, which takes `../data/dataset.csv` and preprocesses it, saving the result at `../data/preprocessed_dataset.csv`:

   ```bash
   python preprocess.py
   ```

2. **Data Augmentation**
   - Run `augmentation.py`, which takes `../data/preprocessed_dataset.csv` and stores the augmented dataset at `../data/augmented_dataset.csv`:

   ```bash
   python augmentation.py
   ```

3. **Visualization**
   - Run `visualization.py`, which compares the original and augmented datasets and stores the results in `../graphs/visualization`:

   ```bash
   python visualization.py
   ```

4. **Model Training**
   - Run `model.py`, which outputs results to `../graphs/model_output`:

   ```bash
   python model.py
   ```

5. **Optimization**
   - Run `optimization.py`, which stores the results in `../data/modelperformance-metrics` and `../data/model_best_params.csv`:

   ```bash
   python optimization.py
   ```

6. **Metaheuristic Searches**
   - Run the following scripts for different metaheuristic searches:
   - **Particle Swarm Optimization:**
   ```bash
   python pso.py
   ```
   - **Simulated Annealing:**
   ```bash
   python sma.py
   ```
   - **Genetic Algorithm:**
   ```bash
   python ga.py
   ```

7. **Comparison of Results**
   - Run `compare.py`, which outputs `hgbrt_all_searcher.png`:

   ```bash
   python compare.py
   ```

8. **Hyperparameters**
   - Run `params.py` to see what hyperparameters have been chosen by each method:

   ```bash
   python params.py
   ```


