# ML 101 Homework 3: Logistic Regression

In this assignment we will implement logistic regression using only numpy.

## Implementation

You should implement not implemented methods of `logistic_regression.py` and `gradient_descent.py` file. Fix "FIXME"s and replace NotImplementedErrors by implementations of corresponding functions.

### LogisticRegression

The model to fit classified data and predict on the new data.

#### _generate_initial_weights

Initialize weights somehow (it would be better, if the initialization was random.)

#### loss

Calculates logistic regression loss on the given data.


#### gradloss

Calculates the gradient of loss function respect to model's weights.

#### hesianloss

Calculates the hessian matrix of loss function respect to model's weights. It is used in Newton-Raphson method.

#### calculate_probabilities

You should calculate <img src="https://render.githubusercontent.com/render/math?math=P(C_1 | \phi)"> for <img src="https://render.githubusercontent.com/render/math?math=\phi"> vectors in the given data.

#### predict

You should return the predicted classes: 0 or 1. Most probably you should use calculate_probabilities function.

### Gradient Descent

You should implement stochastic, minibatch, batch gradient descents. You might want to shuffle data first for stochastic and minibatch gradient descents. Batch gradient descent fucntion should yield the gradient only once. So should do Newton Raphson method's function(Already implemented).
Note, that gradient descent function should yield updates, not weight vectors. So after gradient descent yields $u$, the model should update weights like $w^{(new)} = w^{(old)} - u$.

### Perceptron (Optional)

You should replace weight initialization, implement calculations of loss function, gradient of the loss function and predict method. Implementation is similar to LogisticRegression model.

## Testing

After implementing everything, please run unit tests and make sure your code passes all of them.

```
python -m pytest
```

Run `plot.py` to see the training process of the model. You can choose the model, solution method and some extra parameters, e.g. `python plot.py --update-method stochastic_gradient_descent --num-datapoints 10`. Use `python plot.py --help` to see the full list of instructions.
