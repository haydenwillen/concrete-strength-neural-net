# Predicting Concrete Compressive Strength with Neural Networks

**Hayden Willen** | Machine Learning II | University of Virginia School of Data Science

This project predicts the compressive strength of concrete (in MPa) from its mix ingredients and curing age, using 1,030 lab-tested mixes. It follows I-Cheng Yeh's [1998 paper](https://doi.org/10.1016/S0008-8846(98)00165-3) on modeling high-performance concrete with neural networks and compares a PyTorch neural network against scikit-learn's version, a support vector machine, and linear regression.

Strength is normally measured by curing a sample for 28 days and crushing it, so a model that predicts strength from the mix can save weeks of testing and wasted material.

## What's in the notebook

1. **Analysis plan.** How we planned the exploration, preprocessing, model setup and cross-validation before writing code.
2. **Exploratory data analysis.** Data quality checks, distributions, each input plotted against strength, strength by curing age, and a correlation heatmap.
3. **Neural network in PyTorch.** An `nn.Module` network (8 → 64 → 32 → 1, ReLU) trained with MSE loss, Adam, and a `DataLoader`, scored with 5-fold cross-validation.
4. **Replication with scikit-learn.** The same network rebuilt with `MLPRegressor` on the same folds.
5. **Model comparison.** An RBF kernel SVR and linear regression on the same folds, compared on MSE, RMSE, MAE, and R².

## Results

Mean scores across 5 folds. The best in each column is in bold.

| Model | Test MSE | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|
| Neural network (scikit-learn) | **29.79** | **5.45** | **3.91** | **0.892** |
| Neural network (PyTorch) | 30.58 | 5.51 | 3.96 | 0.890 |
| SVR (RBF kernel) | 34.90 | 5.90 | 4.12 | 0.874 |
| Linear regression | 111.01 | 10.51 | 8.34 | 0.601 |

All models use the same 8 inputs and the same fold splits. Errors are in MPa (MSE in MPa²).

## Key findings

- **Non-linear models win by a wide margin.** Both neural networks have about a quarter of the error of linear regression. Strength depends on ingredient combinations and rises then levels off with age, which a straight line cannot capture.
- **PyTorch and scikit-learn agree.** With the same architecture and training settings, the two networks land within about 1 MSE of each other, well inside the fold-to-fold spread.
- **SVR is a close second.** The RBF kernel also handles the curved relationships, with an R² of 0.874 against 0.89 for the networks.
- **Cement, age, and water matter most.** Cement has the strongest positive correlation with strength (0.50), and water the strongest negative one (-0.29). Strength climbs fast in the first 28 days and gains little after about 56 days.
- **Seeding once is not enough for reproducibility.** Setting the PyTorch seed only at the top gave different numbers on each rerun, so the training loop reseeds at the start of every fold.

## Data

The [Concrete Compressive Strength dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) comes from the UCI Machine Learning Repository and was donated by I-Cheng Yeh, the author of the paper above. It is included in `data/Concrete_Data.xls` under the CC BY 4.0 license, and all credit for the data goes to the original author.

## How to run

```bash
git clone https://github.com/haydenwillen/concrete-strength-neural-net.git
cd concrete-strength-neural-net
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook concrete_strength_neural_net.ipynb
```

## Project structure

```
concrete-strength-neural-net/
├── concrete_strength_neural_net.ipynb   # full analysis with outputs
├── data/
│   └── Concrete_Data.xls                # UCI dataset, 1,030 mixes
├── requirements.txt                     # Python packages
├── .gitignore                           # files Git should skip
└── README.md
```

## Limitations

- **Duplicate rows.** The data has 25 duplicate mixes. We kept them since repeated lab tests are realistic, but a duplicate can land in both a training and a test fold, which may slightly inflate scores.
- **No tuning.** Every model uses fixed settings by design, so none of these scores are the best each model could do.
- **Convergence warnings.** `MLPRegressor` warns that it had not fully converged after 300 epochs. We kept 300 to match the PyTorch network.
- **Fold-to-fold variation.** The PyTorch network's MSE varies by about 5.4 across folds, so small gaps between the top models are not meaningful.
- **Different setup from the paper.** The paper used its own network design and train/test splits, so these numbers are not a direct replication of its results.

## Credits

This started as a group lab for Machine Learning II at the University of Virginia. For this version, I reviewed the full notebook, made the results reproducible, added a findings section, and rewrote it as a standalone project. I used Claude to help with the cleanup.
