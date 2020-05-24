## Setup

[Read the complete tutorial here](https://www.curiousily.com/posts/reproducible-machine-learning-and-experiment-tracking-pipiline-with-python-and-dvc/)

```
git clone git@github.com:curiousily/Reproducible-ML-with-DVC.git
```

```
pipenv install --dev
```

```
git checkout pre-dvc
```

## DVC

Initialize DVC

```
dvc init
```

and add remote storage (local in this case)

```
dvc remote add -d localremote /tmp/dvc-storage
```

disable analytics (optional)

```
dvc config core.analytics false
```

## Experiment with Linear Regression

Build Dataset

```
dvc run -f assets/data.dvc \
    -d studentpredictor/create_dataset.py \
    -o assets/data \
    python studentpredictor/create_dataset.py
```

Create features

```
dvc run -f assets/features.dvc \
    -d studentpredictor/create_features.py \
    -d assets/data \
    -o assets/features \
    python studentpredictor/create_features.py
```

Train model

```
dvc run -f assets/models.dvc \
    -d studentpredictor/train_model.py \
    -d assets/features \
    -o assets/models \
    python studentpredictor/train_model.py
```

Evaluate the model and save metrics (RMSE and r^2)

```
dvc run -f assets/evaluate.dvc \
    -d studentpredictor/evaluate_model.py \
    -d assets/features \
    -d assets/models \
    -M assets/metrics.json \
    python studentpredictor/evaluate_model.py
```

Check the metrics for your current model:

```sh
dvc metrics show -T
```

## Experiment with Random Forest

Checkout the Random Forest experiment:

```
git checkout rf-experiment
```

Reproduce everything with the RF model

```
dvc repro assets/evaluate.dvc
```

Check the metrics for the Random Forest model compared to the Linear Regression:

```sh
dvc metrics show -T
```

[Read the complete tutorial here](https://www.curiousily.com/posts/reproducible-machine-learning-and-experiment-tracking-pipiline-with-python-and-dvc/)

## License

MIT
