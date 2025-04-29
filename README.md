## Simple Recurrent Neural Network (RNN) from Scratch

### Objective:
Constructed a basic Recurrent Neural Network (RNN) from first principles using NumPy, with the aim of understanding temporal sequence modeling, recurrent computation, and backpropagation through time without relying on deep learning frameworks. Compared performance against TensorFlow’s built-in `SimpleRNN` to validate correctness.

### Key Features:

- **Manual Architecture Implementation:** Developed core RNN components by hand, including forward and backward passes, hidden state propagation, and autoregressive prediction for future time steps.
  
- **Training Logic:** Built a training loop from scratch using Mean Squared Error (MSE) loss and gradient descent. Implemented backpropagation through time (BPTT) to optimize parameters.

- **Benchmarking:** Trained a TensorFlow RNN with an equivalent architecture on the same synthetic time series data, enabling direct performance comparison.

- **Prediction Visualization:** Plotted predicted sequences from both models against ground truth to assess temporal accuracy.

### Skills Demonstrated:

- Applied core deep learning concepts (recurrent layers, BPTT) using only NumPy.
- Benchmarked custom model against TensorFlow for reliability and performance alignment.
- Used autoregressive logic for multi-step forecasting.
- Employed Python class structures for modular, testable design.

### Result:
Both the custom RNN and the TensorFlow model produced closely aligned predictions on noisy linear data, achieving comparable MSE and MAE scores. This confirmed the correctness and robustness of the hand-crafted implementation.

![Model Predictions](/assets/Figure_1.png)
