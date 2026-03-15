import csv
import os
import time
import math
import tempfile
from pympler import asizeof
from .HyperLogLog import HyperLogLog

class BenchmarkEngine:
    def __init__(self, standard_error=0.03, max_items=100000, step_size=5000):
        self.standard_error = standard_error
        self.max_items = max_items
        self.step_size = step_size

    def _get_hll_memory(self, hll_instance):
        return asizeof.asizeof(hll_instance)

    def _get_set_memory(self, set_instance):
        return asizeof.asizeof(set_instance)

    def run_insertion_benchmark(self, filename="insertion_results.csv"):
        filepath = os.path.join(tempfile.gettempdir(), filename)
        print(f"Running Insertion Benchmark... saving to {filepath}")
        
        # Initialize locally for a clean state
        hll = HyperLogLog(standardError=self.standard_error)
        hash_set = set()
        
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Items", "True_Count", "HLL_Estimate", "Error_Pct", "HLL_Memory_B", "Set_Memory_B", "Theoretical_Error"])
            
            theoretical_err = 1.04 / math.sqrt(hll.numberOfRegisters)

            for i in range(1, self.max_items + 1):
                item = f"unique_session_identifier_string_{i}"
                hll.insertElem(item)
                hash_set.add(item)

                if i % self.step_size == 0:
                    true_count = len(hash_set)
                    hll_est = hll.getCardinality()
                    error_pct = abs(hll_est - true_count) / true_count * 100
                    
                    writer.writerow([i, true_count, hll_est, error_pct, self._get_hll_memory(hll), self._get_set_memory(hash_set), theoretical_err])
                    print(f"  Processed {i} items... Error: {error_pct:.2f}%")

    def run_sparse_dense_benchmark(self, filename="sparse_dense_results.csv"):
        filepath = os.path.join(tempfile.gettempdir(), filename)
        print(f"\nRunning Sparse-to-Dense Transition Benchmark... saving to {filepath}")
        
        hll = HyperLogLog(standardError=0.01) 
        
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Items", "HLL_Memory_B", "Is_Sparse"])

            for i in range(1, 8001): 
                hll.insertElem(f"sparse_test_{i}")
                
                if i % 25 == 0:
                    writer.writerow([i, self._get_hll_memory(hll), hll.isSparse])

    def run_error_rate_benchmark(self, filename="error_rates_results.csv"):
        filepath = os.path.join(tempfile.gettempdir(), filename)
        print(f"\nRunning Multi-Error Rate Comparison... saving to {filepath}")
        
        errors = [0.01, 0.03, 0.10] 
        hlls = {err: HyperLogLog(standardError=err) for err in errors}
        hash_set = set()
        
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = ["Items", "True_Count", "Est_0.01", "Mem_0.01", "Est_0.03", "Mem_0.03", "Est_0.10", "Mem_0.10"]
            writer.writerow(headers)

            for i in range(1, self.max_items + 1):
                item = f"err_test_{i}"
                hash_set.add(item)
                for hll in hlls.values():
                    hll.insertElem(item)
                
                if i % self.step_size == 0:
                    row = [i, len(hash_set)]
                    for err in errors:
                        row.extend([hlls[err].getCardinality(), self._get_hll_memory(hlls[err])])
                    writer.writerow(row)

    def run_merge_benchmark(self, filename="merge_results.csv"):
        filepath = os.path.join(tempfile.gettempdir(), filename)
        print("\nRunning Merge/Union Benchmark...")
        hll1 = HyperLogLog(self.standard_error)
        hll2 = HyperLogLog(self.standard_error)
        set1 = set()
        set2 = set()

        for i in range(1, 60000):
            item = f"item_{i}"
            hll1.insertElem(item)
            set1.add(item)
            
        for i in range(40000, 100000):
            item = f"item_{i}"
            hll2.insertElem(item)
            set2.add(item)

        start_time = time.perf_counter()
        set1.update(set2)
        set_merge_time = (time.perf_counter() - start_time) * 1000 
        set_mem = self._get_set_memory(set1)

        start_time = time.perf_counter()
        hll1.merge(hll2)
        hll_merge_time = (time.perf_counter() - start_time) * 1000 
        hll_mem = self._get_hll_memory(hll1)

        print(f"  Set Update Time: {set_merge_time:.3f} ms | Final Size: {set_mem / 1024 / 1024:.2f} MB")
        print(f"  HLL Merge Time: {hll_merge_time:.3f} ms | Final Size: {hll_mem / 1024:.2f} KB")

        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Structure", "Merge_Time_ms", "Memory_Bytes"])
            writer.writerow(["HashSet", set_merge_time, set_mem])
            writer.writerow(["HyperLogLog", hll_merge_time, hll_mem])