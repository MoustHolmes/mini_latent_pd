import torch
import torch.nn as nn


class EulerSolver(nn.Module):
    """A simple first-order Euler method ODE solver."""

    @torch.no_grad()
    def solve(
        self, model: nn.Module, x0: torch.Tensor, labels: torch.Tensor, steps: int
    ) -> torch.Tensor:
        """Solves the ODE from t=0 to t=1.

        Args:
            model (nn.Module): The model predicting the vector field.
            x0 (torch.Tensor): The initial condition at t=0 (e.g., noise).
            labels (torch.Tensor): The conditional labels.
            steps (int): The number of discretization steps.

        Returns:
            torch.Tensor: The solution at t=1.
        """
        device = x0.device

        num_samples = x0.shape[0]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels).long()
        labels = labels.to(device)

        x = x0.to(device)

        dt = 1.0 / steps

        for i in range(steps):
            t = torch.ones(num_samples, device=device) * i * dt

            # Predict velocity
            velocity = model(x, t, labels)

            # Update step
            x = x + velocity * dt

        return x


class RK4Solver(nn.Module):
    """Fourth-order Runge-Kutta (RK4) ODE solver."""

    @torch.no_grad()
    def solve(
        self, model: nn.Module, x0: torch.Tensor, labels: torch.Tensor, steps: int
    ) -> torch.Tensor:
        """Solves the ODE from t=0 to t=1 using the RK4 method.

        Args:
            model (nn.Module): The model predicting the vector field.
            x0 (torch.Tensor): The initial condition at t=0 (e.g., noise).
            labels (torch.Tensor): The conditional labels.
            steps (int): The number of discretization steps.

        Returns:
            torch.Tensor: The solution at t=1.
        """
        device = x0.device

        num_samples = x0.shape[0]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels).long()
        labels = labels.to(device)

        x = x0.to(device)

        dt = 1.0 / steps

        for i in range(steps):
            t = torch.ones(num_samples, device=device) * i * dt

            # RK4 steps
            k1 = model(x, t, labels)
            k2 = model(x + 0.5 * dt * k1, t + 0.5 * dt, labels)
            k3 = model(x + 0.5 * dt * k2, t + 0.5 * dt, labels)
            k4 = model(x + dt * k3, t + dt, labels)

            # Update step
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return x
