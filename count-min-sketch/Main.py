import cv2
import os
import shutil
import sys
from faker import Faker
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from tqdm import tqdm
import seaborn as sns
from .CountMinSketch import CountMinSketch
from common.IPV4ExperimentData import ExperimentData

class Main:
    def __init__(self, experimentData, numHashFuncs=3, 
                 minWidth=5, maxWidth=5000, iterationStepSize=25, 
                 directoryPath="/tmp/output_cms"):
        
        self.data = experimentData
        self.actualCounts = self.data.get_actual_counts()
        
        self.numHashFuncs = numHashFuncs
        self.widthRange = {"min": minWidth, "max": maxWidth, "step": iterationStepSize}
        self.directoryPath = directoryPath
        
        self.outputs = []
        self.filterStates = []
        
        if os.path.exists(self.directoryPath):
            shutil.rmtree(self.directoryPath)
        os.makedirs(self.directoryPath)

    def _runIterations(self):
        for width in range(self.widthRange["min"], self.widthRange["max"], self.widthRange["step"]):
            cms = CountMinSketch(self.numHashFuncs, width)
            for ip in self.data.inputSet:
                cms.insertElem(ip)
            
            estimates = {ip: cms.getFrequency(ip) for ip in set(self.data.inputSet)}
            self.outputs.append({"width": width, "estimates": estimates})
            self.filterStates.append({"width": width, "state": cms.filter})

    def _saveOutputGraph(self, outputItem):
        df = pd.DataFrame({
            "actualCounts": pd.Series(self.actualCounts),
            "estimatedCounts": pd.Series(outputItem["estimates"])
        })
        error = ((df["estimatedCounts"] - df["actualCounts"]) / df["actualCounts"]).mean()
        
        plt.figure(figsize=(12, 6))
        df.plot(kind='bar', ax=plt.gca(), title=f"Width: {outputItem['width']}, Mean Error: {error:.4f}")
        
        savePath = Path(self.directoryPath) / str(outputItem["width"])
        savePath.mkdir(parents=True, exist_ok=True)
        plt.savefig(savePath / "actual_vs_estimate.png")
        plt.close('all') 
        return df

    def _saveFilterStateGraph(self, filterStateItem):
        plt.figure(figsize=(15, 4))
        sns.heatmap(filterStateItem["state"], cbar_kws={'orientation': 'horizontal'}, cmap='viridis')
        plt.title(f"CMS State (Rows: {self.numHashFuncs}, Width: {filterStateItem['width']})")
        
        savePath = Path(self.directoryPath) / str(filterStateItem["width"])
        savePath.mkdir(parents=True, exist_ok=True)
        plt.savefig(savePath / "filter_state.png", bbox_inches='tight')
        plt.close('all')

    def _createVid(self, frameFileName, fps=5):
        target_size = (1280, 720)
        video_name = os.path.join(self.directoryPath, f'cms_{frameFileName.split(".")[0]}_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_name, fourcc, fps, target_size)

        for op in self.outputs:
            path = os.path.join(self.directoryPath, str(op["width"]), frameFileName)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, target_size)
                out.write(img)
        out.release()
        print(f"Video saved to: {video_name}")

    def run(self):
        print(f"Running CMS Iterations... (Output: {self.directoryPath})")
        self._runIterations()
        
        for item in tqdm(self.filterStates, desc="Saving Filter States"):
            self._saveFilterStateGraph(item)
        
        for item in tqdm(self.outputs, desc="Saving Estimates"):
            df = self._saveOutputGraph(item)
            df.to_csv(Path(self.directoryPath) / str(item["width"]) / "output.csv")
            
        self._createVid("filter_state.png")
        self._createVid("actual_vs_estimate.png")

if __name__ == "__main__":
    
    print("Count-Min Sketch: zipf Distribution")
    exp_data = ExperimentData(dataSetSize=100, inputSetSize=500, distribution='zipf')
    m = Main(exp_data, directoryPath="/tmp/output_cms_zipf")
    m.run()
    
    print("Count-Min Sketch: random Distribution")
    exp_data = ExperimentData(dataSetSize=100, inputSetSize=500, distribution='random')
    m = Main(exp_data, directoryPath="/tmp/output_cms_random")
    m.run()