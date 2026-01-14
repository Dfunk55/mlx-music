"""
Flow Matching Scheduler for ACE-Step.

Implements the Euler, Heun, and PingPong schedulers
for flow matching diffusion.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx


@dataclass
class SchedulerOutput:
    """Output from a scheduler step."""

    prev_sample: mx.array
    pred_original_sample: Optional[mx.array] = None


class FlowMatchEulerDiscreteScheduler:
    """
    Flow Match Euler Discrete Scheduler.

    Implements the flow matching scheduler with Euler method
    for solving the ODE: dx/dt = v(x, t)
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max

        # Initialize timesteps
        self.timesteps: Optional[mx.array] = None
        self.sigmas: Optional[mx.array] = None

        # Set default
        self.set_timesteps(50)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[str] = None,
    ):
        """
        Set the timesteps for inference.

        Args:
            num_inference_steps: Number of denoising steps
            device: Ignored (MLX handles device automatically)
        """
        # Linear timesteps from 1 to near 0
        timesteps = mx.linspace(1.0, 1.0 / self.num_train_timesteps, num_inference_steps)

        # Apply shift transformation
        sigmas = self.shift * timesteps / (1 + (self.shift - 1) * timesteps)

        self.timesteps = timesteps * self.num_train_timesteps
        self.sigmas = sigmas
        self.num_inference_steps = num_inference_steps

    def scale_noise(
        self,
        sample: mx.array,
        timestep: mx.array,
        noise: mx.array,
    ) -> mx.array:
        """
        Scale and add noise to sample based on timestep.

        For flow matching: x_t = (1 - sigma) * x_0 + sigma * noise
        """
        # Get sigma for this timestep
        step_idx = mx.argmin(mx.abs(self.timesteps - timestep))
        sigma = self.sigmas[step_idx]

        # Expand sigma for broadcasting
        while sigma.ndim < sample.ndim:
            sigma = sigma[..., None]

        noisy_sample = (1 - sigma) * sample + sigma * noise
        return noisy_sample

    def step(
        self,
        model_output: mx.array,
        timestep: mx.array,
        sample: mx.array,
        return_dict: bool = True,
    ) -> SchedulerOutput:
        """
        Perform one denoising step.

        For flow matching with Euler:
        x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * v(x_t, t)

        Args:
            model_output: Predicted velocity from the model
            timestep: Current timestep
            sample: Current noisy sample
            return_dict: Whether to return SchedulerOutput

        Returns:
            SchedulerOutput with denoised sample
        """
        # Find current step index
        step_idx = mx.argmin(mx.abs(self.timesteps - timestep))

        # Get current and previous sigma
        sigma = self.sigmas[step_idx]

        if step_idx + 1 < self.num_inference_steps:
            sigma_prev = self.sigmas[step_idx + 1]
        else:
            sigma_prev = mx.array(0.0)

        # Expand for broadcasting
        while sigma.ndim < sample.ndim:
            sigma = sigma[..., None]
            sigma_prev = sigma_prev[..., None]

        # Euler step: x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * velocity
        dt = sigma_prev - sigma
        prev_sample = sample + dt * model_output

        if return_dict:
            return SchedulerOutput(prev_sample=prev_sample)
        return (prev_sample,)


class FlowMatchHeunDiscreteScheduler:
    """
    Flow Match Heun Discrete Scheduler.

    Uses Heun's method (improved Euler) for more accurate
    ODE solving at the cost of 2x model evaluations.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max

        self.timesteps: Optional[mx.array] = None
        self.sigmas: Optional[mx.array] = None

        self.set_timesteps(50)

    def set_timesteps(self, num_inference_steps: int, device: Optional[str] = None):
        """Set timesteps for inference."""
        timesteps = mx.linspace(1.0, 1.0 / self.num_train_timesteps, num_inference_steps)
        sigmas = self.shift * timesteps / (1 + (self.shift - 1) * timesteps)

        self.timesteps = timesteps * self.num_train_timesteps
        self.sigmas = sigmas
        self.num_inference_steps = num_inference_steps

    def scale_noise(
        self,
        sample: mx.array,
        timestep: mx.array,
        noise: mx.array,
    ) -> mx.array:
        """Scale and add noise to sample."""
        step_idx = mx.argmin(mx.abs(self.timesteps - timestep))
        sigma = self.sigmas[step_idx]

        while sigma.ndim < sample.ndim:
            sigma = sigma[..., None]

        return (1 - sigma) * sample + sigma * noise

    def step(
        self,
        model_output: mx.array,
        timestep: mx.array,
        sample: mx.array,
        model_fn: Optional[callable] = None,
        return_dict: bool = True,
    ) -> SchedulerOutput:
        """
        Perform one Heun step.

        Heun's method:
        1. k1 = v(x_t, t)
        2. x_euler = x_t + dt * k1
        3. k2 = v(x_euler, t-1)
        4. x_{t-1} = x_t + dt * (k1 + k2) / 2

        Note: Requires model_fn for the second evaluation.
        If model_fn is None, falls back to Euler.
        """
        step_idx = mx.argmin(mx.abs(self.timesteps - timestep))

        sigma = self.sigmas[step_idx]
        if step_idx + 1 < self.num_inference_steps:
            sigma_prev = self.sigmas[step_idx + 1]
        else:
            sigma_prev = mx.array(0.0)

        while sigma.ndim < sample.ndim:
            sigma = sigma[..., None]
            sigma_prev = sigma_prev[..., None]

        dt = sigma_prev - sigma

        # First step (Euler prediction)
        k1 = model_output
        x_euler = sample + dt * k1

        # If no model function provided, use Euler
        if model_fn is None:
            if return_dict:
                return SchedulerOutput(prev_sample=x_euler)
            return (x_euler,)

        # Second evaluation at predicted point
        if step_idx + 1 < self.num_inference_steps:
            next_timestep = self.timesteps[step_idx + 1]
        else:
            next_timestep = mx.array(0.0)

        k2 = model_fn(x_euler, next_timestep)

        # Heun combination
        prev_sample = sample + dt * (k1 + k2) / 2

        if return_dict:
            return SchedulerOutput(prev_sample=prev_sample)
        return (prev_sample,)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int,
) -> Tuple[mx.array, int]:
    """
    Retrieve timesteps from scheduler.

    Args:
        scheduler: The scheduler instance
        num_inference_steps: Number of denoising steps

    Returns:
        Tuple of (timesteps array, num_inference_steps)
    """
    scheduler.set_timesteps(num_inference_steps)
    return scheduler.timesteps, num_inference_steps


def get_scheduler(
    scheduler_type: str = "euler",
    num_train_timesteps: int = 1000,
    shift: float = 3.0,
):
    """
    Get scheduler by type.

    Args:
        scheduler_type: "euler" or "heun"
        num_train_timesteps: Number of training timesteps
        shift: Shift parameter for flow matching

    Returns:
        Scheduler instance
    """
    if scheduler_type.lower() == "euler":
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )
    elif scheduler_type.lower() == "heun":
        return FlowMatchHeunDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
