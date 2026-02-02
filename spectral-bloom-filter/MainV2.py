import sys
from faker import Faker
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import seaborn as sns
from .SpectralBloomFilter import SpectralBloomFilter
from common.IPV4ExperimentData import ExperimentData
from common.miscFunctions import create_video_from_dir

class MainV2:
    def __init__(self, numHashFuncs=3, width=25,
                 minInputSetSize=100, maxInputSetSize=5000, iterationStepSize=100, 
                 distribution='random', directoryPath="/tmp/output_sbf_v2"):
        
        self.numHashFuncs = numHashFuncs
        self.width = width
        self.inputSetSizeRange = {"min": minInputSetSize, "max": maxInputSetSize, "step": iterationStepSize}
        self.distribution = distribution
        self.directoryPath = directoryPath
        
        self.outputs = []
        self.filterStates = []
        
        if os.path.exists(self.directoryPath):
            shutil.rmtree(self.directoryPath)
        os.makedirs(self.directoryPath)

    def _runIterations(self):
        for inputSetSize in tqdm(range(self.inputSetSizeRange["min"], self.inputSetSizeRange["max"], self.inputSetSizeRange["step"]), desc="Running Iterations"):
            
            exp_data = ExperimentData(dataSetSize=100, inputSetSize=inputSetSize, distribution=self.distribution)
            actualCounts = exp_data.get_actual_counts()
            
            sbf = SpectralBloomFilter(self.numHashFuncs, self.width)
            
            currentFilterStateItem = {"inputSetSize": inputSetSize, "state": []}
            count = 0
            for ip in tqdm(exp_data.inputSet, desc="Inserting IP into SBF"):
                sbf.insertElem(ip)
                count+=1
                currentFilterStateItem = {"inputSetSize": count, "state": sbf.filter}
                self._saveCurrentFilterStateGraph(currentFilterStateItem, inputSetSize)
            
            estimates = {ip: sbf.getFrequency(ip) for ip in set(exp_data.inputSet)}
            self.outputs.append({"inputSetSize": inputSetSize, "estimates": estimates, "actualCounts": actualCounts})
            self.filterStates.append(currentFilterStateItem)

    def _saveOutputGraph(self, outputItem):
        df = pd.DataFrame({
            "actualCounts": pd.Series(outputItem["actualCounts"]),
            "estimatedCounts": pd.Series(outputItem["estimates"])
        })
        error = ((df["estimatedCounts"] - df["actualCounts"]) / df["actualCounts"]).mean()
        
        plt.figure(figsize=(12, 6))
        df.plot(kind='bar', ax=plt.gca(), title=f"Input Size: {outputItem['inputSetSize']}, Mean Error: {error:.4f}")
        
        savePath = Path(self.directoryPath) / str(outputItem["inputSetSize"])
        savePath.mkdir(parents=True, exist_ok=True)
        plt.savefig(savePath / "actual_vs_estimate.png")
        plt.close('all')
        return df
        
    def _saveCurrentFilterStateGraph(self, filterStateItem, parentIterationSize):
        state = np.array(filterStateItem["state"]).reshape(1, -1)
        plt.figure(figsize=(15, 4))
        plt.imshow(state, aspect='auto', cmap='Wistia')
        plt.colorbar(orientation='horizontal')
        plt.title(f"SBF State (Width: {self.width}, Input Size: {filterStateItem['inputSetSize']})")
        
        savePath = Path(self.directoryPath) / str(parentIterationSize) / "frames"
        savePath.mkdir(parents=True, exist_ok=True)
        plt.savefig(savePath / f"filter_state_frame{filterStateItem['inputSetSize']}.png", bbox_inches='tight')
        plt.close('all')

    def run(self):
        print(f"Running SBF Iterations... (Output: {self.directoryPath})")
        self._runIterations()
        
        for item in tqdm(self.outputs, desc="Saving Estimates"):
            df = self._saveOutputGraph(item)
            df.to_csv(Path(self.directoryPath) / str(item["inputSetSize"]) / "output.csv")

if __name__ == "__main__":
    
    inputSize = 5000
    
    print("Spectral Bloom Filter (V2): zipf Distribution")
    dirPath = "/tmp/output_sbf_v2_zipf"
    m_zipf = MainV2(distribution='zipf', directoryPath=dirPath, minInputSetSize=inputSize, maxInputSetSize=inputSize+1, iterationStepSize=100)
    m_zipf.run()
    create_video_from_dir(f"{dirPath}/{inputSize}/frames")
    
    print("Spectral Bloom Filter (V2): random Distribution")
    dirPath = "/tmp/output_sbf_v2_random"
    m_random = MainV2(distribution='random', directoryPath=dirPath, minInputSetSize=inputSize, maxInputSetSize=inputSize+1, iterationStepSize=100)
    m_random.run()
    create_video_from_dir(f"{dirPath}/{inputSize}/frames")
