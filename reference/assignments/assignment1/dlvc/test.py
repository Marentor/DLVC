from abc import ABCMeta, abstractmethod

import numpy
import numpy as np
import torch


class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass

    @abstractmethod
    def __lt__(self, other) -> bool:
        '''
        Return true if this performance measure is worse than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass

    @abstractmethod
    def __gt__(self, other) -> bool:
        '''
        Return true if this performance measure is better than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass


class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self):
        '''
        Ctor.
        '''

        self.target = []
        self.prediction = []
        self.reset()

    def reset(self):
        '''
        Resets the internal state.
        '''
        self.target.clear()
        self.prediction.clear()

    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
            The predicted class label is the one with the highest probability.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''
        predictions = numpy.argmax(prediction, 1)
        self.prediction.append(predictions)
        self.target.append(target)

    def __str__(self):
        '''
        Return a string representation of the performance.
        '''
        acc= self.accuracy()
        s=f"accuracy: {acc:.3f}"
        return s
        # return something like "accuracy: 0.395"

    def __lt__(self, other) -> bool:
        '''
        Return true if this accuracy is worse than another one.
        Raises TypeError if the types of both measures differ.
        '''

        # See https://docs.python.org/3/library/operator.html for how these
        # operators are used to compare instances of the Accuracy class
        if self.accuracy() < other:
            return True
        else:
            return False

    def __gt__(self, other) -> bool:
        '''
        Return true if this accuracy is better than another one.
        Raises TypeError if the types of both measures differ.
        '''

        if self.accuracy() > other:
            return True
        else:
            return False

    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''

        correct = 0
        size = 0
        if self.target is None or self.prediction is None:
            return 0
        else:
            for i in range(len(self.prediction)):
                correct += (self.prediction[i] == self.target[i]).sum()
                size += len(self.prediction[i])
        return correct / size
