# Model Card


## Model Details

- Algorithm: Logistic Regression

- Hyperparameters:
- Regularization: L2
- Solver: liblinear
- Maximum Iterations: 100

## Intended Use

Predict if a given individual have an income higher or lower than 50k.

## Training Data

Random subset of the "Adult dataset" publicly available at
https://archive.ics.uci.edu/dataset/20/census+income.

## Evaluation Data

Remaining subset after training data split.

## Metrics

fbeta: 0.7162
precision = 0.2615
recall = 0.3831

## Ethical Considerations

The data is non personally identifiable.

## Caveats and Recommendations
Model might drift using categories that are not continuous as the encode wouldn't
represent them properly.
