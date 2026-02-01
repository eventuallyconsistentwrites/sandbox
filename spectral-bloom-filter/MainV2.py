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
        for inputSetSize in range(self.inputSetSizeRange["min"], self.inputSetSizeRange["max"], self.inputSetSizeRange["step"]):
            
            exp_data = ExperimentData(dataSetSize=100, inputSetSize=inputSetSize, distribution=self.distribution)
            actualCounts = exp_data.get_actual_counts()
            
            sbf = SpectralBloomFilter(self.numHashFuncs, self.width)
            for ip in exp_data.inputSet:
                sbf.insertElem(ip)
            
            estimates = {ip: sbf.getFrequency(ip) for ip in set(exp_data.inputSet)}
            self.outputs.append({"inputSetSize": inputSetSize, "estimates": estimates, "actualCounts": actualCounts})
            self.filterStates.append({"inputSetSize": inputSetSize, "state": sbf.filter})

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

    def _saveFilterStateGraph(self, filterStateItem):
        state = np.array(filterStateItem["state"]).reshape(1, -1)
        plt.figure(figsize=(15, 4))
        plt.imshow(state, aspect='auto', cmap='Wistia')
        plt.colorbar(orientation='horizontal')
        plt.title(f"SBF State (Width: {self.width}, Input Size: {filterStateItem['inputSetSize']})")
        
        savePath = Path(self.directoryPath) / str(filterStateItem["inputSetSize"])
        savePath.mkdir(parents=True, exist_ok=True)
        plt.savefig(savePath / "filter_state.png", bbox_inches='tight')
        plt.close('all')

    def _createVid(self, frameFileName, fps=5):
        target_size = (1280, 720)
        video_name = os.path.join(self.directoryPath, f'sbf_{frameFileName.split(".")[0]}_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_name, fourcc, fps, target_size)

        for op in self.outputs:
            path = os.path.join(self.directoryPath, str(op["inputSetSize"]), frameFileName)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, target_size)
                out.write(img)
        out.release()
        print(f"Video saved to: {video_name}")

    def run(self):
        print(f"Running SBF Iterations... (Output: {self.directoryPath})")
        self._runIterations()
        
        for item in tqdm(self.filterStates, desc="Saving Filter States"):
            self._saveFilterStateGraph(item)
        
        for item in tqdm(self.outputs, desc="Saving Estimates"):
            df = self._saveOutputGraph(item)
            df.to_csv(Path(self.directoryPath) / str(item["inputSetSize"]) / "output.csv")
            
        self._createVid("filter_state.png")
        self._createVid("actual_vs_estimate.png")

if __name__ == "__main__":
    
    print("Spectral Bloom Filter (V2): zipf Distribution")
    m_zipf = MainV2(distribution='zipf', directoryPath="/tmp/output_sbf_v2_zipf")
    m_zipf.run()
    
    print("Spectral Bloom Filter (V2): random Distribution")
    m_random = MainV2(distribution='random', directoryPath="/tmp/output_sbf_v2_random")
    m_random.run()
