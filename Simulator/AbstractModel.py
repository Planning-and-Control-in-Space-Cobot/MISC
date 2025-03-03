from abc import ABC, abstractmethod
import numpy as np

"""
This File Contains the Abstract Class Model that will be used to define the
 interface for the models that will be passed to the simulator class.

In order to get the dynamics of the system correctly the user should inherit
 form this class, and overwrite at least the constructo and the f function.
"""


class Model(ABC):
    """This is an abstract class that defines the interface for the model that
    will be passed to the simulator class."""

    @abstractmethod
    def __init__(self, **kwargs):
        """This is the default constructor of the class, this need to be
        overwritten since there different model will required different
        arguments, meaning we want to be able to have custom constructors for
        each model."""
        self.kwargs = kwargs
        pass

    @abstractmethod
    def f(self, t: float, state: np.ndarray, u: np.ndarray):
        """This function returns the time derivative of the state given the
        current state and the input u as well as the time t.

        Take into consideration the solver will assume the u is constant during
          this period of time 0 to t

        Arguments:
            t (float): Time (May be required for the computation of the time derivative)
            state (np.array): The current state of the system
            u (np.array): The input to the system

        Returns:
            np.array: The time derivative of the state, This should be a 1D array,
            meaning all the derivatives should be flattend and then stacked.
        """
        pass

    def validate_state(self, state: np.ndarray):
        """This function will validate the state and make sure that the state is
        in the correct format, if not it will raise an error.

        Arguments:
            state (np.array): The state of the system

        Returns:
            Bool: True if the state is valid, False otherwise
        """

        if not isinstance(state, np.ndarray):
            raise ValueError("The state should be a numpy array")

        if len(state.shape) != 1:
            raise ValueError("The state should be a 1D array")

        return True

    def validate_input(self, u: np.ndarray):
        """This function will validade the input and make sure it is in the
        correct format, if not it will raise an error.

        Argumetns:
            u (np.array): The input to the system

        Returns:
            Bool: True if the input is valid, False otherwise
        """

        if not isinstance(u, np.ndarray):
            raise ValueError("The input should be a numpy array")

        if len(u.shape) != 1:
            raise ValueError("The input should be a 1D array")

        return True
