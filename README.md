# Linear Regression from Scratch

### The Basic Mathematical Intuition

The simplest equation of linear equation is $y = mx + b$ where $m$ is the slope and $b$ is the intercept of the straight line.

Now the error($E$) can be calculated as $E = {\frac 1 n}\sum_{i=0}^{n} (y_i - (mx_i+b))^2$ where $y_i$ is the actual value of the data point at $i$ location.
<br>This is the **Mean Squared Error(MSE)**.

### Gradient Descent

The partial derivatives of the error with respect to $m$ and $b$ indicate the direction of steepest ascent- the direction in which the error increases the fastest.
<br>Since our goal in gradient descent is to minimize the error, we take the negative of these derivatives. This gives us the direction of steepest descent, allowing us to update $m$ and $b$ in a way that reduces the error.

Now, partially derivating $E$ with respect to $m$, we get,

$$
\frac{\partial E}{\partial m} = \frac{1}{n} \sum_{i=0}^{n} 2(y_i - (mx_i + b))(-x_i) = -\frac{2}{n} \sum_{i=0}^{n} x_i(y_i - (mx_i + b))
$$


Similarly partially derivating $E$ with respect to $b$, we get,

$$
\frac{\partial E}{\partial b} = \frac{1}{n} \sum_{i=0}^{n} 2(y_i - (mx_i + b))(-1) = -\frac{2}{n} \sum_{i=0}^{n} (y_i - (mx_i + b))
$$


Finally the last thing is to update the value of $m$ and $b$ at each iteration taking small steps, known as the $learning \ rate  (L)$ by the following way:

$$
m=m-L*\frac{\partial E}{\partial m}
$$
$$
b=b-L*\frac{\partial E}{\partial b}
$$

### Implementation

The linear regression algorithm is implemented from scratch in Python, encapsulated within a `LinearRegression` class. This approach emphasizes object-oriented design and makes the model easy to maintain, extend, and reuse.

Here's a breakdown of the implementation:

- **Data Handling**: The model takes a Pandas `DataFrame` containing `x` and `y` values. These are the independent and dependent variables, respectively.
- **Initialization**: The constructor initializes the slope `m`, intercept `b`, learning rate, and the number of training epochs.
- **Loss Function**: Computes the Mean Squared Error (MSE) between predicted and actual values over the entire dataset.
- **Gradient Descent**: Updates the model parameters `m` and `b` using the gradients of the loss function. The update is performed iteratively based on the learning rate to minimize the loss.
- **Training with Visualization**:
  - Trains the model for a specified number of epochs.
  - In each epoch, it:
      - Calculates and prints the current loss.
      - Updates the parameters using gradient descent.
      - Dynamically visualizes the regression line's progression using matplotlib.
  - The plot shows both the data points and the regression line evolving during training, helping to intuitively understand convergence.

This implementation avoids machine learning libraries like `Scikit-learn`. Only essential libraries — `Pandas` and `Matplotlib` are used for data handling and plotting.

### About the dataset

The file `data.csv` used in this implementation contains randomly generated synthetic data. The data simulates a linear relationship between variables `x` and `y` with some added noise, which is typical in real-world regression problems.

This synthetic dataset allows us to:
- Visualize and understand how linear regression fits a line through noisy data.
- Evaluate the learning process clearly without relying on external or complex datasets.
- Focus on the mechanics of gradient descent and model optimization.

The dataset includes two columns:
- `x`: Independent variable
- `y`: Dependent variable (with some randomness to mimic real-world variability)
