import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from scipy.spatial import ConvexHull
from functools import partial


result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def generate_data(n_samples=100):
    np.random.seed(0)
    radial_dist = np.sqrt(np.random.rand(n_samples))
    angle_dist = 2 * np.pi * np.random.rand(n_samples)
    features = np.c_[radial_dist * np.cos(angle_dist), radial_dist * np.sin(angle_dist)]
    labels = ((radial_dist > 0.5) * 2 - 1).reshape(-1, 1)
    return features, labels

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(42)
        self.lr = lr
        self.activation_fn = activation

        self.weights_ih = np.random.randn(input_dim, hidden_dim) * 0.1
        self.bias_hidden = np.zeros((1, hidden_dim))
        self.weights_ho = np.random.randn(hidden_dim, output_dim) * 0.1
        self.bias_output = np.zeros((1, output_dim))

        self.hidden_values = None
        self.grad_vals = None

    def forward(self, inputs):
        hidden_layer_input = np.dot(inputs, self.weights_ih) + self.bias_hidden
        if self.activation_fn == 'tanh':
            self.hidden_values = np.tanh(hidden_layer_input)
        elif self.activation_fn == 'relu':
            self.hidden_values = np.maximum(0, hidden_layer_input)
        elif self.activation_fn == 'sigmoid':
            self.hidden_values = 1 / (1 + np.exp(-hidden_layer_input))
        output_layer_input = np.dot(self.hidden_values, self.weights_ho) + self.bias_output
        return np.tanh(output_layer_input)

    def backward(self, inputs, labels):
        output_error = self.forward(inputs) - labels
        d_output = output_error * (1 - self.forward(inputs)**2)

        grad_weights_ho = np.dot(self.hidden_values.T, d_output)
        grad_bias_output = np.sum(d_output, axis=0, keepdims=True)

        hidden_error = np.dot(d_output, self.weights_ho.T)
        if self.activation_fn == 'tanh':
            d_hidden = hidden_error * (1 - self.hidden_values**2)
        elif self.activation_fn == 'relu':
            d_hidden = hidden_error * (self.hidden_values > 0)
        elif self.activation_fn == 'sigmoid':
            d_hidden = hidden_error * (self.hidden_values * (1 - self.hidden_values))

        grad_weights_ih = np.dot(inputs.T, d_hidden)
        grad_bias_hidden = np.sum(d_hidden, axis=0, keepdims=True)

        self.weights_ih -= self.lr * grad_weights_ih
        self.bias_hidden -= self.lr * grad_bias_hidden
        self.weights_ho -= self.lr * grad_weights_ho
        self.bias_output -= self.lr * grad_bias_output

        self.grad_vals = {
            'input_hidden': grad_weights_ih,
            'hidden_output': grad_weights_ho
        }

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    hidden_features = mlp.hidden_values
    scatter = ax_hidden.scatter(
        hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
        c=y.ravel(), cmap='bwr', alpha=0.7
    )
    ax_hidden.set_title(f"Hidden Space at Step {frame * 10}")

    x_vals = np.linspace(-1.5, 1.5, 50)
    y_vals = np.linspace(-1.5, 1.5, 50)
    xx, yy = np.meshgrid(x_vals, y_vals)
    z_vals = -(mlp.weights_ho[0, 0] * xx +
               mlp.weights_ho[1, 0] * yy +
               mlp.bias_output[0, 0]) / (mlp.weights_ho[2, 0] + 1e-5)
    ax_hidden.plot_surface(xx, yy, z_vals, alpha=0.3, color='tan')

    if hidden_features.shape[1] >= 3:
        try:
            hull = ConvexHull(hidden_features)
            for simplex in hull.simplices:
                ax_hidden.plot_trisurf(
                    hidden_features[simplex, 0], hidden_features[simplex, 1], hidden_features[simplex, 2],
                    color='blue', alpha=0.2, shade=True
                )
        except Exception as e:
            print(f"ConvexHull Error: {e}")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).reshape(xx.shape)

    ax_input.contour(xx, yy, preds, levels=[0], colors='black', linewidths=1.5)
    ax_input.contourf(xx, yy, preds, levels=[-1, 0, 1], colors=['red', 'blue'], alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k', s=20)
    ax_input.set_title(f"Input Space at Step {frame * 10}")

    ax_gradient.set_title(f"Gradients at Step {frame * 10}")

    node_positions = {
        'x1': (0.0, 0.0), 'x2': (0.0, 1.0),
        'h1': (0.5, 0.3), 'h2': (0.5, 0.7), 'h3': (0.5, 1.0),
        'y': (1.0, 0.7)
    }

    for name, (x, y) in node_positions.items():
        ax_gradient.add_patch(Circle((x, y), 0.05, color='blue'))
        offset = 0.05
        if name.startswith('x'):
            ax_gradient.text(x - offset, y, name, color='black', ha='right')
        elif name.startswith('y'):
            ax_gradient.text(x + offset, y, name, color='black', ha='left')
        else:
            ax_gradient.text(x, y + offset, name, color='black', ha='center')

    edges = [
        ('x1', 'h1', mlp.grad_vals['input_hidden'][0, 0]),
        ('x1', 'h2', mlp.grad_vals['input_hidden'][0, 1]),
        ('x1', 'h3', mlp.grad_vals['input_hidden'][0, 2]),
        ('x2', 'h1', mlp.grad_vals['input_hidden'][1, 0]),
        ('x2', 'h2', mlp.grad_vals['input_hidden'][1, 1]),
        ('x2', 'h3', mlp.grad_vals['input_hidden'][1, 2]),
        ('h1', 'y', mlp.grad_vals['hidden_output'][0, 0]),
        ('h2', 'y', mlp.grad_vals['hidden_output'][1, 0]),
        ('h3', 'y', mlp.grad_vals['hidden_output'][2, 0]),
    ]

    for start, end, grad in edges:
        x1, y1 = node_positions[start]
        x2, y2 = node_positions[end]
        line_width = max(0.5, min(5, abs(grad) * 10))
        ax_gradient.plot([x1, x2], [y1, y2], 'm-', linewidth=line_width)


    return [scatter]

def visualize(activation, lr, step_num):
    features, labels = generate_data()
    nn_model = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    fig = plt.figure(figsize=(21, 7))
    axes = [
        fig.add_subplot(131, projection='3d'),
        fig.add_subplot(132),
        fig.add_subplot(133),
    ]

    anim = FuncAnimation(
        fig,
        partial(update, mlp=nn_model, ax_input=axes[1], ax_hidden=axes[0], ax_gradient=axes[2], X=features, y=labels),
        frames=step_num // 10,
        repeat=False
    )

    anim.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()


if __name__ == "__main__":
    visualize(activation="tanh", lr=0.1, step_num=1000)
