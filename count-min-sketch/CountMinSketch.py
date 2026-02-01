import mmh3

class CountMinSketch:
    def __init__(self, numHashFuncs, width):
        self.numHashFuncs = numHashFuncs
        self.width = width
        self.seeds = list(range(1, self.numHashFuncs + 1))
        self.filter = [[0] * self.width for _ in range(self.numHashFuncs)]

    def _getElementPositions(self, elem):
        return [mmh3.hash(elem, seed) % self.width for seed in self.seeds]

    def insertElem(self, elem):
        positions = self._getElementPositions(elem)
        for i, pos in enumerate(positions):
            self.filter[i][pos] += 1

    def getFrequency(self, elem):
        positions = self._getElementPositions(elem)
        return min(self.filter[i][pos] for i, pos in enumerate(positions))