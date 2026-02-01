import mmh3

class SpectralBloomFilter:
    def __init__(self, numHashFuncs, width):
        self.numHashFuncs = numHashFuncs
        self.width = width
        self.seeds = list(range(1, self.numHashFuncs + 1))
        self.filter = [0] * self.width

    def _getElementPositions(self, elem):
        return [mmh3.hash(elem, seed) % self.width for seed in self.seeds]

    def insertElem(self, elem):
        positions = self._getElementPositions(elem)
        # SBF optimization: only increment minimums
        minVal = min(self.filter[pos] for pos in positions)
        for position in positions:
            if self.filter[position] == minVal:
                self.filter[position] += 1

    def getFrequency(self, elem):
        positions = self._getElementPositions(elem)
        return min(self.filter[pos] for pos in positions)