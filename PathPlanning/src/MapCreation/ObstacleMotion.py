from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.transform import Rotation as R

class Motion(ABC):
    """Abstract base class for all motion models."""

    @abstractmethod
    def __call__(self, t: float) -> np.ndarray:
        """Return position at time t."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

class NoMotion(Motion):
    """Static position, no motion."""
    def __init__(self, position=np.zeros(3)):
        """Initialize with a fixed position.
        
        Args:
            position (array-like): Fixed position in 3D space.
        """
        self.position = np.array(position)

    def __call__(self, t: float) -> np.ndarray:
        """Return the fixed position regardless of time."""
        return self.position

    def __repr__(self) -> str:
        return f"NoMotion(position={self.position.tolist()})"

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

class AttitudeMotion(ABC):
    """Abstract base class for all attitude motion models."""

    @abstractmethod
    def __call__(self, t: float) -> R:
        """Return orientation (as a Rotation object) at time t."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

class NoAttitudeMotion(AttitudeMotion):
    """Static orientation, no attitude change."""
    def __init__(self, orientation: R = R.identity()):
        self.orientation = orientation

    def __call__(self, t: float) -> R:
        return self.orientation

    def __repr__(self) -> str:
        return f"NoAttitudeMotion(orientation={self.orientation.as_quat().tolist()})"

class CircleAttitudeMotion(AttitudeMotion):
    """Attitude rotates around a fixed axis at constant angular speed."""
    def __init__(self, speed: float = 1.0, axis=np.array([0, 0, 1])):
        """
        Args:
            speed (float): Angular speed in radians per second.
            axis (array-like): Rotation axis in 3D space.
        """
        self.speed = speed
        self.axis = np.array(axis) / np.linalg.norm(axis)

    def __call__(self, t: float) -> R:
        angle = self.speed * t
        return R.from_rotvec(angle * self.axis)

    def __repr__(self) -> str:
        return f"CircleAttitudeMotion(speed={self.speed}, axis={self.axis.tolist()})"

class LinearAttitudeMotion(AttitudeMotion):
    """Attitude interpolates linearly between two orientations."""
    def __init__(self, start: R, end: R, duration: float):
        """
        Args:
            start (R): Initial orientation.
            end (R): Final orientation.
            duration (float): Time duration over which interpolation occurs.
        """
        self.start = start
        self.end = end
        self.duration = duration
        self.slerp = R.slerp(0, 1, [self.start, self.end])  # Pre-compute slerp object

    def __call__(self, t: float) -> R:
        alpha = np.clip(t / self.duration, 0.0, 1.0)
        return self.slerp(alpha)

    def __repr__(self) -> str:
        return f"LinearAttitudeMotion(start={self.start.as_quat().tolist()}, end={self.end.as_quat().tolist()}, duration={self.duration})"

class SineAttitudeMotion(AttitudeMotion):
    """Oscillates around a base orientation with sinusoidal angular offset."""
    def __init__(self, axis=np.array([0, 1, 0]), frequency=1.0, amplitude=0.5, base: R = R.identity()):
        """
        Args:
            axis (array-like): Axis around which to oscillate.
            frequency (float): Frequency of oscillation.
            amplitude (float): Max angular deviation in radians.
            base (R): Base orientation around which to oscillate.
        """
        self.axis = np.array(axis) / np.linalg.norm(axis)
        self.frequency = frequency
        self.amplitude = amplitude
        self.base = base

    def __call__(self, t: float) -> R:
        angle = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        oscillation = R.from_rotvec(angle * self.axis)
        return self.base * oscillation  # Apply oscillation after base orientation

    def __repr__(self) -> str:
        return f"SineAttitudeMotion(axis={self.axis.tolist()}, frequency={self.frequency}, amplitude={self.amplitude}, base={self.base.as_quat().tolist()})"