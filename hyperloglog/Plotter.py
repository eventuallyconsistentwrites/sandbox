import csv
import os
import tempfile
import matplotlib.pyplot as plt

class BenchmarkPlotter:
    def __init__(self):
        # Color palette inspired by the reference images
        self.C_TEAL = '#4EB1A3'     
        self.C_ORANGE = '#F17D23'   
        self.C_GREY = '#4A4A4A'     
        self.C_RED = '#E05D5D'      
        self.C_PURPLE = '#9B59B6'   

        # Apply global styles
        plt.rcParams.update({
            'font.family': ['Comic Sans MS', 'Humor Sans', 'sans-serif'], 
            'text.color': self.C_GREY,
            'axes.labelcolor': self.C_GREY,
            'axes.edgecolor': self.C_GREY,
            'xtick.color': self.C_GREY,
            'ytick.color': self.C_GREY,
            'axes.titleweight': 'bold'
        })

    def _get_filepath(self, filename):
        """Helper method to dynamically resolve the temp directory path"""
        return os.path.join(tempfile.gettempdir(), filename)

    def plot_insertion_results(self, filename="insertion_results.csv"):
        items, true_counts, hll_estimates, hll_mems, set_mems, theor_errs = [], [], [], [], [], []

        with open(self._get_filepath(filename), mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                items.append(int(row["Items"]))
                tc = int(row["True_Count"])
                true_counts.append(tc)
                hll_estimates.append(float(row["HLL_Estimate"]))
                hll_mems.append(int(row["HLL_Memory_B"]) / (1024 * 1024))
                set_mems.append(int(row["Set_Memory_B"]) / (1024 * 1024))
                theor_errs.append(float(row["Theoretical_Error"]))

        # --- Graph 1: Memory Comparison Plot ---
        plt.figure("Memory Comparison", figsize=(8, 5))
        plt.plot(items, set_mems, label="HashSet Memory (MB)", color=self.C_RED, linewidth=2)
        plt.plot(items, hll_mems, label="HLL Memory (MB)", color=self.C_TEAL, linewidth=2)
        plt.title("Deep Memory Consumption (Pympler)")
        plt.xlabel("Number of Insertions")
        plt.ylabel("Memory in Megabytes (MB)")
        plt.legend()
        plt.grid(color='#E5E5E5', linestyle='--', linewidth=1)
        plt.tight_layout()

        # --- Graph 2: Accuracy Plot with Standard Error Bounds ---
        plt.figure("Estimation Accuracy", figsize=(8, 5))
        plt.plot(items, true_counts, label="Actual Count", color=self.C_GREY, linestyle='dashed', linewidth=2)
        plt.plot(items, hll_estimates, label="HLL Estimate", color=self.C_PURPLE, alpha=0.9, linewidth=2)
        
        upper_1se = [tc + (tc * err) for tc, err in zip(true_counts, theor_errs)]
        lower_1se = [tc - (tc * err) for tc, err in zip(true_counts, theor_errs)]
        upper_2se = [tc + (tc * err * 2) for tc, err in zip(true_counts, theor_errs)]
        lower_2se = [tc - (tc * err * 2) for tc, err in zip(true_counts, theor_errs)]

        plt.fill_between(items, lower_1se, upper_1se, color=self.C_PURPLE, alpha=0.3, label='±1 Std Error (68%)')
        plt.fill_between(items, lower_2se, upper_2se, color=self.C_PURPLE, alpha=0.1, label='±2 Std Errors (95%)')

        plt.title("Cardinality Estimation with Statistical Bounds")
        plt.xlabel("Number of Insertions")
        plt.ylabel("Unique Item Count")
        plt.legend()
        plt.grid(color='#E5E5E5', linestyle='--', linewidth=1)
        plt.tight_layout()

    def plot_sparse_dense(self, filename="sparse_dense_results.csv"):
        items, mems, is_sparse = [], [], []
        with open(self._get_filepath(filename), mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                items.append(int(row["Items"]))
                mems.append(int(row["HLL_Memory_B"]) / 1024) 
                is_sparse.append(row["Is_Sparse"] == 'True')

        transition_item = next((item for item, sparse in zip(items, is_sparse) if not sparse), None)

        # --- Graph 3: Sparse to Dense Transition ---
        plt.figure("Sparse to Dense Transition", figsize=(8, 5))
        plt.plot(items, mems, label="HLL Memory (KB)", color=self.C_TEAL, linewidth=2)
        if transition_item:
            plt.axvline(x=transition_item, color=self.C_ORANGE, linestyle='--', linewidth=2, label=f'Converted to Dense (~{transition_item} items)')
        
        plt.title("HyperLogLog Sparse-to-Dense Memory Profile")
        plt.xlabel("Number of Insertions")
        plt.ylabel("Memory in Kilobytes (KB)")
        plt.legend()
        plt.grid(color='#E5E5E5', linestyle='--', linewidth=1)
        plt.tight_layout()

    def plot_error_rates(self, filename="error_rates_results.csv"):
        items, tc = [], []
        est_01, mem_01, est_03, mem_03, est_10, mem_10 = [], [], [], [], [], []

        with open(self._get_filepath(filename), mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                items.append(int(row["Items"]))
                tc.append(int(row["True_Count"]))
                est_01.append(float(row["Est_0.01"]))
                mem_01.append(int(row["Mem_0.01"]) / 1024)
                est_03.append(float(row["Est_0.03"]))
                mem_03.append(int(row["Mem_0.03"]) / 1024)
                est_10.append(float(row["Est_0.10"]))
                mem_10.append(int(row["Mem_0.10"]) / 1024)

        # --- Graph 4: Multi-Error Rate Accuracy ---
        plt.figure("Error Rates Accuracy", figsize=(8, 5))
        plt.plot(items, tc, label="Actual Count", color=self.C_GREY, linestyle='dashed', linewidth=2)
        plt.plot(items, est_01, label="HLL (1% Error)", color=self.C_TEAL, alpha=0.9, linewidth=2)
        plt.plot(items, est_03, label="HLL (3% Error)", color=self.C_ORANGE, alpha=0.9, linewidth=2)
        plt.plot(items, est_10, label="HLL (10% Error)", color=self.C_RED, alpha=0.9, linewidth=2)
        plt.title("Accuracy Comparison Across Standard Errors")
        plt.xlabel("Number of Insertions")
        plt.ylabel("Estimated Unique Items")
        plt.legend()
        plt.grid(color='#E5E5E5', linestyle='--', linewidth=1)
        plt.tight_layout()

        # --- Graph 5: Multi-Error Rate Memory ---
        plt.figure("Error Rates Memory", figsize=(8, 5))
        plt.plot(items, mem_01, label="HLL Memory (1% Error)", color=self.C_TEAL, linewidth=2)
        plt.plot(items, mem_03, label="HLL Memory (3% Error)", color=self.C_ORANGE, linewidth=2)
        plt.plot(items, mem_10, label="HLL Memory (10% Error)", color=self.C_RED, linewidth=2)
        plt.title("The Memory Cost of Precision (KB)")
        plt.xlabel("Number of Insertions")
        plt.ylabel("Memory in Kilobytes (KB)")
        plt.legend()
        plt.grid(color='#E5E5E5', linestyle='--', linewidth=1)
        plt.tight_layout()

    def plot_merge_results(self, filename="merge_results.csv"):
        structures, times, mems = [], [], []
        with open(self._get_filepath(filename), mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                structures.append(row["Structure"])
                times.append(float(row["Merge_Time_ms"]))
                mems.append(float(row["Memory_Bytes"]) / (1024 * 1024))

        # --- Graph 6: Merge Time ---
        plt.figure("Merge Operations Time", figsize=(6, 5))
        plt.bar(structures, times, color=[self.C_ORANGE, self.C_TEAL], edgecolor=self.C_GREY)
        plt.title("Merge / Update Time (ms)")
        plt.ylabel("Milliseconds")
        plt.tight_layout()

        # --- Graph 7: Merge Final Memory ---
        plt.figure("Merge Final Memory", figsize=(6, 5))
        plt.bar(structures, mems, color=[self.C_ORANGE, self.C_TEAL], edgecolor=self.C_GREY)
        plt.title("Post-Merge Memory Footprint (MB)")
        plt.ylabel("Megabytes (MB)")
        plt.tight_layout()

    def plot_intersection_results(self, filename="intersection_results.csv"):
        overlap_pcts, true_intersections, hll_intersections, error_pcts = [], [], [], []
        
        with open(self._get_filepath(filename), mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                overlap_pcts.append(float(row["Overlap_Pct"]))
                true_intersections.append(float(row["True_Intersection"]))
                hll_intersections.append(float(row["HLL_Intersection"]))
                error_pcts.append(float(row["Error_Pct"]))
                
        # --- Graph 8: The Intersection Trap (Error Compounding) ---
        plt.figure("The Intersection Trap", figsize=(8, 5))
        plt.plot(overlap_pcts, error_pcts, marker='o', color=self.C_RED, linewidth=2, label="Estimation Error %")
        plt.title("The Intersection Trap: Inclusion-Exclusion Error Compounding")
        plt.xlabel("Set Overlap Percentage (%)")
        plt.ylabel("Intersection Estimation Error (%)")
        plt.gca().invert_xaxis() 
        plt.legend()
        plt.grid(color='#E5E5E5', linestyle='--', linewidth=1)
        plt.tight_layout()

        # --- Graph 9: Intersection Volumes Comparison ---
        plt.figure("Intersection Volume Comparison", figsize=(8, 5))
        x_labels = [f"{pct}%" for pct in overlap_pcts]
        x = range(len(x_labels))
        width = 0.35
        
        plt.bar([pos - width/2 for pos in x], true_intersections, width, label='True Intersection', color=self.C_GREY)
        plt.bar([pos + width/2 for pos in x], hll_intersections, width, label='HLL Estimate', color=self.C_ORANGE)
        
        plt.title("Intersection Cardinality: True vs Estimated")
        plt.xlabel("Set Overlap Percentage")
        plt.ylabel("Number of Distinct Elements")
        plt.xticks(x, x_labels)
        plt.legend()
        plt.grid(color='#E5E5E5', linestyle='--', linewidth=1, axis='y')
        plt.tight_layout()

        # --- Graph 10: Signal vs. Noise (Error Proportion) ---
        plt.figure("Signal vs Noise Proportion", figsize=(8, 5))
        
        true_props = []
        error_props = []
        
        # Calculate the relative composition of True Signal vs Absolute Error
        for true_val, est_val in zip(true_intersections, hll_intersections):
            abs_err = abs(est_val - true_val)
            total_magnitude = true_val + abs_err
            
            if total_magnitude > 0:
                true_props.append((true_val / total_magnitude) * 100)
                error_props.append((abs_err / total_magnitude) * 100)
            else:
                true_props.append(0)
                error_props.append(0)
                
        plt.bar(x, true_props, label='True Intersection (Signal)', color=self.C_TEAL, edgecolor=self.C_GREY)
        # Stack the error bar on top of the true value bar
        plt.bar(x, error_props, bottom=true_props, label='Absolute Error (Noise)', color=self.C_RED, edgecolor=self.C_GREY)
        
        plt.title("Signal vs. Noise: Error Proportion at Shrinking Overlaps")
        plt.xlabel("Set Overlap Percentage")
        plt.ylabel("Proportion of Total Magnitude (%)")
        plt.xticks(x, x_labels)
        
        # Add a 50% threshold line to show exactly when noise becomes larger than signal
        plt.axhline(50, color=self.C_GREY, linestyle='dotted', linewidth=1.5)
        plt.text(0.5, 52, "50% Noise Threshold", color=self.C_GREY)
        
        # Reverse legend order so Noise displays on top of Signal, matching the bar stack
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(reversed(handles), reversed(labels), loc='upper left')
        
        plt.tight_layout()

    def show_all(self):
        """Displays all generated figures concurrently."""
        plt.show()