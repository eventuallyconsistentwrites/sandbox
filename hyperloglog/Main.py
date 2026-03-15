from .Benchmark import BenchmarkEngine
from .Plotter import BenchmarkPlotter

class Main:
    def __init__(self):
        self.insertion_csv = "insertion_results.csv"
        self.merge_csv = "merge_results.csv"
        self.sparse_dense_csv = "sparse_dense_results.csv"
        self.error_rates_csv = "error_rates_results.csv"
        self.intersection_csv = "intersection_results.csv"

    def execute(self):
        print("==========================================")
        print("  HyperLogLog Benchmarking Suite Started  ")
        print("==========================================\n")
        
        # 1. Initialize benchmark engine
        engine = BenchmarkEngine(standard_error=0.03, max_items=100000, step_size=5000)
        
        # 2. Run all benchmarks
        engine.run_insertion_benchmark(filename=self.insertion_csv)
        engine.run_sparse_dense_benchmark(filename=self.sparse_dense_csv)
        engine.run_error_rate_benchmark(filename=self.error_rates_csv)
        engine.run_merge_benchmark(filename=self.merge_csv)
        engine.run_intersection_benchmark(filename=self.intersection_csv)
        
        print("\nBenchmarking Complete! Compiling visualizer data...")

        # 3. Queue the plots
        plotter = BenchmarkPlotter()
        plotter.plot_insertion_results(filename=self.insertion_csv)
        plotter.plot_sparse_dense(filename=self.sparse_dense_csv)
        plotter.plot_error_rates(filename=self.error_rates_csv)
        plotter.plot_merge_results(filename=self.merge_csv)
        plotter.plot_intersection_results(filename=self.intersection_csv)
        
        # 4. Show all windows at once
        print("Launching windows...")
        plotter.show_all()

if __name__ == "__main__":
    app = Main()
    app.execute()