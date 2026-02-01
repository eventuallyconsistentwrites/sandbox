from faker import Faker
import random
import numpy as np

class ExperimentData:
    def __init__(self, dataSetSize=100, inputSetSize=500, distribution='zipf', alpha=1.2):
        self.dataSetSize = dataSetSize
        self.inputSetSize = inputSetSize
        self.distribution = distribution
        self.alpha = alpha
        self.fake = Faker()
        self.dataSet = []
        self.inputSet = []
        self._generate_data()

    def _generate_data(self):
        # Generate the pool of data (e.g., all possible IPs)
        self.dataSet = [self.fake.ipv4() for _ in range(self.dataSetSize)]
        
        # Generate the stream/input based on distribution
        if self.distribution == 'zipf':
            samples = np.random.zipf(self.alpha, self.inputSetSize) - 1
            # Modulo ensures we stay within the dataset bounds
            self.inputSet = [self.dataSet[s % self.dataSetSize] for s in samples]
        elif self.distribution == 'random':
            self.inputSet = random.choices(self.dataSet, k=self.inputSetSize)
        else:
            raise ValueError("Distribution must be 'zipf' or 'random'")
            
    def get_actual_counts(self):
        return {ip: self.inputSet.count(ip) for ip in set(self.inputSet)}