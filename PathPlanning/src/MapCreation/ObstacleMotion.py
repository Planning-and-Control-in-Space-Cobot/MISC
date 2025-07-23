from abc import ABC, abstractmethod
import numpy as np

class Motion(ABC):
    """Abstract base class for all motion models."""

    @abstractmethod
    def __call__(self, t: float) -> np.ndarray:
        """Return position at time t."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

class CircleMotion(Motion):
    """Circular motion model"""
    def __init__(self, radius=1.0, speed=1.0, center=np.zeros(3)):
        """Initialize circular motion parameters.
        
        Args: 
            radius (float): Radius of the circular path.
            speed (float): Angular speed in radians per second.
            center (array-like): Center of the circle in 3D space.
        
        Returns:
            None
        """
        self.radius = radius
        self.speed = speed
        self.center = np.array(center)

    def __call__(self, t: float) -> np.ndarray:
        """Calculate position at time t.

        Args:
            t (float): Time in seconds.
        Returns:
            np.ndarray: Position in 3D space at time t.
        """

        angle = self.speed * t
        return self.center + self.radius * np.array([np.cos(angle), np.sin(angle), 0])

    def __repr__(self) -> str:
        """String representation of the CircleMotion instance.
        
        Returns:
            str: A string describing the CircleMotion instance.
        
        Example:
            CircleMotion(radius=1.0, speed=1.0, center=[0.0, 0.0, 0.0])
        """
        return f"CircleMotion(radius={self.radius}, speed={self.speed}, center={self.center.tolist()})"

class LinearMotion(Motion):
    def __init__(self, direction, speed, start=np.zeros(3)):
        """Initialize linear motion parameters.
        
        Args:
            direction (array-like): Direction vector for the motion.
            speed (float): Speed of the motion.
            start (array-like): Starting position in 3D space.
        
        Returns:
            None
        """
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.speed = speed
        self.start = np.array(start)

    def __call__(self, t: float) -> np.ndarray:
        """Calculate position at time t.

        Args:
            t (float): Time in seconds.
        Returns:
            np.ndarray: Position in 3D space at time t.
        """
        return self.start + self.direction * self.speed * t

    def __repr__(self) -> str:
        """String representation of the LinearMotion instance.

        Returns:
            str: A string describing the LinearMotion instance.
        """
        return f"LinearMotion(direction={self.direction.tolist()}, speed={self.speed}, start={self.start.tolist()})"

class SineMotion(Motion):
    def __init__(self, amplitude=1.0, frequency=1.0, axis=[0, 0, 1], offset=np.zeros(3)):
        """Initialize sine wave motion parameters.
        
        Args:
            amplitude (float): Amplitude of the sine wave.
            frequency (float): Frequency of the sine wave.
            axis (array-like): Axis of oscillation.
            offset (array-like): Offset in 3D space.
        Returns:
            None
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.axis = np.array(axis)
        self.offset = np.array(offset)

    def __call__(self, t: float) -> np.ndarray:
        """Calculate position at time t.

        Args:
            t (float): Time in seconds.
        Returns:
            np.ndarray: Position in 3D space at time t.
        """
        displacement = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        return self.offset + displacement * self.axis

    def __repr__(self) -> str:
        """String representation of the SineMotion instance.
        
        Returns:
            str: A string describing the SineMotion instance.
        """
        return f"SineMotion(amplitude={self.amplitude}, frequency={self.frequency}, axis={self.axis.tolist()}, offset={self.offset.tolist()})"
