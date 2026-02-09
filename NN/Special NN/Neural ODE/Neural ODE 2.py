import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim, augment_dim=1):
        super(ODEFunc, self).__init__()
        self.augment_dim = augment_dim
        self.net = nn.Sequential(
            nn.Linear(2 + self.augment_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 + self.augment_dim)
        )

    def forward(self, t, x):
        return self.net(x)

class ODESolver(nn.Module):
    def __init__(self, func, method='euler', augment_dim=1):
        super(ODESolver, self).__init__()
        self.func = func
        self.method = method
        self.augment_dim = augment_dim

    def forward(self, x0, t):
        h = t[1] - t[0]  # Assuming uniform time steps
        trajectory = [x0]
        x = x0

        for i in range(len(t) - 1):
            if self.method == 'euler':
                # Simple Euler method
                dx = self.func(t[i], x) * h
                x = x + dx
            elif self.method == 'rk4':
                # 4th-order Runge-Kutta method
                k1 = self.func(t[i], x)
                k2 = self.func(t[i] + h/2, x + h/2 * k1)
                k3 = self.func(t[i] + h/2, x + h/2 * k2)
                k4 = self.func(t[i] + h, x + h * k3)
                dx = h/6 * (k1 + 2*k2 + 2*k3 + k4)
                x = x + dx

            trajectory.append(x)

        return torch.stack(trajectory)

def generate_spiral_data(n_points=100, noise=0.1):
    theta = np.linspace(0, 3 * np.pi, n_points)
    r = theta + 1
    x = r * np.cos(theta) + np.random.normal(0, noise, n_points)
    y = r * np.sin(theta) + np.random.normal(0, noise, n_points)
    return np.column_stack((x, y))

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    augment_dim = 1  # Number of augmented dimensions to add

    n_points = 50
    spiral_data = generate_spiral_data(n_points=n_points, noise=0.1)
    spiral_data = torch.tensor(spiral_data, dtype=torch.float32)

    t = torch.linspace(0, 1, n_points)

    func = ODEFunc(hidden_dim=100, augment_dim=augment_dim)
    ode_solver = ODESolver(func, method='euler', augment_dim=augment_dim)

    # Initial condition (starting point of the spiral) with augmented dimensions
    x0_aug = torch.zeros(1, 2 + augment_dim, dtype=torch.float32)
    x0_aug[0, :2] = spiral_data[0]  # Copy original dimensions
    x0 = x0_aug

    optimizer = optim.Adam(func.parameters(), lr=0.01)

    # Training loop
    n_epochs = 300
    losses = []

    for epoch in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        pred_trajectory = ode_solver(x0, t)

        # Loss is MSE between predicted and true trajectory (only for original dimensions)
        loss = nn.MSELoss()(pred_trajectory[:, :, :2].squeeze(), spiral_data)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}')

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    # Plot true vs predicted trajectories
    plt.subplot(1, 3, 2)
    pred_trajectory_np = pred_trajectory.detach().squeeze().numpy()[:, :2]  # Only use original dimensions
    spiral_data_np = spiral_data.numpy()

    plt.scatter(spiral_data_np[:, 0], spiral_data_np[:, 1], label='True', s=10, alpha=0.5)
    plt.plot(pred_trajectory_np[:, 0], pred_trajectory_np[:, 1], 'r-', label='Predicted')
    plt.scatter(x0.detach().numpy()[0, 0], x0.detach().numpy()[0, 1], c='g', s=100, label='Initial Point')
    plt.legend()
    plt.title('True vs Predicted Trajectory')

    # Plot vector field
    plt.subplot(1, 3, 3)

    x = np.linspace(-10, 10, 20)
    y = np.linspace(-10, 10, 20)
    X, Y = np.meshgrid(x, y)

    # Compute vector field - we need to augment these points too
    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
    grid_points_aug = np.zeros((grid_points.shape[0], 2 + augment_dim))
    grid_points_aug[:, :2] = grid_points  # Only set the first 2 dimensions

    grid_points_tensor = torch.tensor(grid_points_aug, dtype=torch.float32)

    with torch.no_grad():
        derivatives = func(0, grid_points_tensor).numpy()

    # We're only interested in visualizing the derivatives of the original dimensions
    U = derivatives[:, 0].reshape(X.shape)
    V = derivatives[:, 1].reshape(X.shape)

    plt.quiver(X, Y, U, V, alpha=0.5)
    plt.scatter(spiral_data_np[:, 0], spiral_data_np[:, 1], label='True', s=10, alpha=0.5)
    plt.title('Vector Field')


    plt.tight_layout()
    plt.show()

    print("Training complete!")


main()