import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf


def xavier(size_in: int, size_out: int) -> np.ndarray:
    """
    Xavier initialization for weight matrices.

    Args:
        size_in: Number of input neurons (fan-in).
        size_out: Number of output neurons (fan-out).

    Returns:
        Initialized weight matrix of shape (size_out, size_in).
    """
    return np.random.randn(size_out, size_in) * np.sqrt(1.0 / size_in)


class BaseRNN:
    """
    Base class for RNN implementations providing common utilities.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def mse_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute mean squared error loss."""
        return np.mean((y_true - y_pred) ** 2)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 500,
        lr: float = 0.01,
        verbose: bool = True,
        print_every: int = 10,
    ) -> None:
        """
        Standard training loop: forward, loss, backward, and parameter update.
        """
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss = self.mse_loss(y, y_pred)
            self.backward(X, y, y_pred, lr)
            if verbose and epoch % print_every == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}")

    
    def predict_next(self, X: np.ndarray, steps: int = 3) -> np.ndarray:
        """
        Autoregressive prediction for future time steps.
        """
        seq = [x.ravel() for x in X]
        preds = []
        for _ in range(steps):
            input_seq = np.array(seq[-len(X):])  # keep same window size
            y_seq = self.forward(input_seq)
            next_val = y_seq[-1]
            preds.append(next_val)
            seq.append(next_val.reshape(self.input_size))
        return np.array(preds).reshape(-1, 1)



class SimpleRNNFromScratch(BaseRNN):
    """
    Simple RNN implemented from first principles using numpy.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ):
        super().__init__(input_size, hidden_size, output_size)
        # Weight and bias initialization
        self.Wh = xavier(input_size + hidden_size, hidden_size)
        self.bh = np.zeros((hidden_size, 1))
        self.Wy = xavier(hidden_size, output_size)
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the entire sequence.

        Args:
            inputs: Array of shape (time_steps, input_size).

        Returns:
            Outputs of shape (time_steps, output_size).
        """
        h = np.zeros((self.hidden_size, 1))
        self.last_inputs = []
        self.last_h = [h.copy()]
        outputs = []

        for x in inputs:
            x = x.reshape(-1, 1)
            combined = np.vstack((h, x))
            h = np.tanh(self.Wh.dot(combined) + self.bh)
            y = self.Wy.dot(h) + self.by
            outputs.append(y.flatten())
            self.last_inputs.append(combined)
            self.last_h.append(h.copy())

        return np.vstack(outputs)

    def backward(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lr: float,
    ) -> None:
        """
        Backward pass through time to update gradient parameters.
        """
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        dWh = np.zeros_like(self.Wh)
        dbh = np.zeros_like(self.bh)
        dh_next = np.zeros((self.hidden_size, 1))

        # Reverse to iterate backwards
        for t in reversed(range(len(X))):
            dy = (y_pred[t].reshape(-1, 1) - y_true[t].reshape(-1, 1))
            dWy += dy.dot(self.last_h[t + 1].T)
            dby += dy

            dh = self.Wy.T.dot(dy) + dh_next
            h_raw = self.last_h[t + 1]
            dtanh = (1 - h_raw ** 2) * dh

            dWh += dtanh.dot(self.last_inputs[t].T)
            dbh += dtanh
            dh_next = self.Wh.T[: self.hidden_size, :].dot(dtanh)

        # Gradient descent parameter update
        for param, grad in zip(
            [self.Wh, self.bh, self.Wy, self.by],
            [dWh, dbh, dWy, dby],
        ):
            param -= lr * grad

def tensorflow_rnn_model(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    input_size: int,
    hidden_size: int,
    output_size: int,
    epochs: int = 300,
) -> tf.keras.Model:
    """
    Build and train a TensorFlow SimpleRNN model on the sequence.

    Args:
        X_seq: Input sequence of shape (time_steps, features).
        y_seq: Target sequence of same shape.
        input_size: Input size for the RNN.
        hidden_size: Hidden layer size.
        output_size: Output size for the final layer.
        epochs: Number of training epochs.

    Returns:
        Trained Keras SimpleRNN model.
    """
    tf.keras.backend.clear_session()
    tf.random.set_seed(39)
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(hidden_size, return_sequences=True, input_shape=(None, input_size)),
        tf.keras.layers.Dense(output_size)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_seq.reshape(1, -1, input_size),
        y_seq.reshape(1, -1, output_size),
        epochs=epochs,
        verbose=0,
        shuffle=False
    )
    return model


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(39)

    # Hyperparameters
    time_steps = 10
    input_size = 1
    hidden_size = 5
    output_size = 1
    rnn_epochs = 500
    tf_epochs = 300

    # Generate synthetic data
    X_seq = np.linspace(0, 1, time_steps).reshape(time_steps, input_size)
    y_seq = 2 * X_seq + 0.1 * np.random.randn(*X_seq.shape)

    # Train custom from-scratch RNN
    simple_rnn = SimpleRNNFromScratch(input_size, hidden_size, output_size)
    simple_rnn.train(X_seq, y_seq, epochs=rnn_epochs, lr=0.01)
    y_pred = simple_rnn.forward(X_seq)
    future_preds = simple_rnn.predict_next(X_seq, steps=3)

    # Train TensorFlow RNN model
    tf_model = tensorflow_rnn_model(
        X_seq,
        y_seq,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        epochs=tf_epochs,
    )
    tf_pred = tf_model.predict(X_seq.reshape(1, -1, 1)).reshape(-1, 1)

    # Compute metrics
    custom_mse = mean_squared_error(y_seq, y_pred)
    custom_mae = mean_absolute_error(y_seq, y_pred)
    tf_mse     = mean_squared_error(y_seq, tf_pred)
    tf_mae     = mean_absolute_error(y_seq, tf_pred)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(y_seq.flatten(), label='True')
    plt.plot(y_pred.flatten(), linestyle='--', label='Custom RNN')
    plt.plot(tf_pred.flatten(), linestyle='-.', label='TF RNN')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Model Predictions vs. True Signal')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Tabular metrics
    df = pd.DataFrame({
        'Model': ['Custom RNN', 'TensorFlow RNN'],
        'MSE':   [custom_mse, tf_mse],
        'MAE':   [custom_mae, tf_mae],
    })
    print("\nMetrics Comparison:\n", df.to_string(index=False))

    # Future predictions (custom)
    print("\nFuture 3 steps (custom):", future_preds.flatten())
