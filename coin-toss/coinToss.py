import os.path
import random
from pathlib import Path

import pandas as pd
import shutil

HEAD = "H"
TAIL = "T"

# Perform a Single Coin Toss
def toss():
    result = random.randint(0, 1)
    if result == 1:
        return HEAD
    return TAIL

# Perform n Coin Tosses
def nTosses(n: int):
    return [toss() for _ in range(n)]

# Perform n experiments each performing m coin tosses
def nExperiments(n: int, m: int):
    return [nTosses(m) for _ in range(n)]

# Perform m experiments for n universes
def nUniverses(n: int, m: int, numberOfCoinTosses: int):
    return [nExperiments(m, numberOfCoinTosses) for _ in range(n)]

def writeExperimentsToFile(experiments, filename="experiments.csv", directory=os.path.join("","tmp","experiments")):
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)

    results = []
    for experiment in experiments:
        countOfLeadingTails = len(experiment)
        if HEAD in experiment:
            countOfLeadingTails = experiment.index(HEAD)
        results.append({"tosses": "".join(experiment), "count": countOfLeadingTails})
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(directory, filename), index=False)

if __name__ == "__main__":
    numberOfUniverses = 6
    numberOfExperiments = 10
    numberOfTosses = 10
    universes = nUniverses(numberOfUniverses, numberOfExperiments, numberOfTosses)
    for i, universe in enumerate(universes):
        writeExperimentsToFile(universe, f"experiment{i}.csv")



