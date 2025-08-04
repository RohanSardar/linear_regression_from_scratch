import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, data: pd.DataFrame, m: float, b: float, learning_rate: float, epochs: int) -> None:
        self.data = data
        self.m = m
        self.b = b
        self.learning_rate = learning_rate
        self.epochs = epochs

    def loss_fn(self) -> float:
        error = 0
        for i in range(len(self.data)):
            x: float = self.data.iloc[i].x
            y: float = self.data.iloc[i].y
            error += (y - (self.m*x + self.b)) ** 2
        return error / float(len(self.data))

    def gradient_descent(self, m_new: float, b_new: float) -> tuple[float, float]:
        m_grad = 0
        b_grad = 0
        n: int = len(self.data)
        for i in range(n):
            x: float = self.data.iloc[i].x
            y: float = self.data.iloc[i].y
            m_grad += (-2/n) * x * (y - (m_new*x + b_new))
            b_grad += (-2/n) * (y - (m_new*x + b_new))
        m: float = m_new - self.learning_rate * m_grad
        b: float = b_new - self.learning_rate * b_grad
        return m, b

    def train_plot(self) -> None:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(self.data.x, self.data.y, color="#2a118f", label="Data Points")

        line, = ax.plot([], [], color="#f29c07", linewidth=2.5, label="Regression Line")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Linear Regression Training")
        ax.legend()

        for epoch in range(self.epochs):
            loss: float = self.loss_fn()
            self.m, self.b = self.gradient_descent(self.m, self.b)

            x_vals: list[float] = list(range(int(self.data.x.min()), int(self.data.x.max()) + 1))
            y_vals: list[float] = [self.m * x + self.b for x in x_vals]

            line.set_data(x_vals, y_vals)

            ax.set_title(f"Epoch {epoch+1}/{self.epochs}   Loss: {loss:.3f}   y = {self.m:.3f}x + {self.b:.3f}",
                        fontsize=13, color="#34495E")
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.3f}, y = {self.m:.3f}x + {self.b:.3f}")

            plt.pause(0.05)

        plt.ioff()
        plt.show()


data: pd.DataFrame = pd.read_csv('data.csv')
m: float = 0
b: float = 0
learning_rate: float = 1e-5
epochs: int = 100

linear_regression: LinearRegression = LinearRegression(data, m, b, learning_rate, epochs)
linear_regression.train_plot()
