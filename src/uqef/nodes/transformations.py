from abc import ABC, abstractmethod
import numpy as np

class TransformationStrategy(ABC):
    """Base class for parameter transformation strategies"""
    
    @abstractmethod
    def transform(self, samples, source_dist, target_dist):
        """Transform samples from source to target distribution"""
        pass

class GeneralTransformation(TransformationStrategy):
    """Uses forward and inverse CDF for any distribution pair"""
    
    def transform(self, samples, source_dist, target_dist):
        """
        https://chaospy.readthedocs.io/en/master/user_guide/advanced_topics/generalized_polynomial_chaos.html
        :param samples: array of samples from source_dist
        :param source_dist: 'standard' distribution
        :param target_dist: 'user-defined' distribution
        :return: array of samples from target_dist
        """
        return target_dist.inv(source_dist.fwd(samples))

class LinearUniformTransformation(TransformationStrategy):
    """Optimized linear transformation for uniform distributions"""
    
    def transform(self, samples, source_dist, target_dist):
        """
        :param samples: array of samples from source_dist, when source_dist is U[-1,1] or U[0,1]
        :param source_dist: 'standard' distribution either U[-1,1] or U[0,1]
        :param target_dist: 'user-defined' distribution
        :return: array of samples from target_dist
        """
        dim = len(source_dist)
        _a = np.empty([dim, 1])
        _b = np.empty([dim, 1])
        
        for i in range(dim):
            r_lower = source_dist[i].lower
            q_lower = target_dist[i].lower
            q_upper = target_dist[i].upper
            
            if r_lower == -1:
                _a[i] = (q_lower + q_upper) / 2
                _b[i] = (q_upper - q_lower) / 2
            elif r_lower == 0:
                _a[i] = q_lower
                _b[i] = (q_upper - q_lower)
        
        return _a + _b * samples

class AdaptiveTransformation(TransformationStrategy):
    """Automatically selects best transformation strategy"""
    
    def __init__(self):
        self.linear_uniform = LinearUniformTransformation()
        self.general = GeneralTransformation()
    
    def transform(self, samples, source_dist, target_dist):
        if self._is_uniform_pair(source_dist, target_dist):
            return self.linear_uniform.transform(samples, source_dist, target_dist)
        return self.general.transform(samples, source_dist, target_dist)
    
    def _is_uniform_pair(self, source_dist, target_dist):
        # Check if both are uniform distributions
        try:
            for dist in source_dist:
                if not hasattr(dist, 'lower'):
                    return False
            return True
        except:
            return False