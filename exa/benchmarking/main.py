import time
import psutil
import os
from termcolor import colored
import platform

class BenchmarkSuite:
    def __init__(self, llm_instance, backend, dtype):
        self.llm = llm_instance
        self.initial_memory = psutil.Process(os.getpid()).memory_info().rss
        self.history = []

        self.backend = backend
        self.dtype = dtype


    def measure_memory(self):
        current_memory = psutil.Process(os.getpid()).memory_info().rss
        return (current_memory - self.initial_memory) / (1024 ** 2)  # Convert to MB

    def measure_throughput(self, input_text):
        durations = []
        for _ in range(5):  # Adaptive sampling: running 5 times
            start_time = time.time()
            self.llm.run(input_text)
            durations.append(time.time() - start_time)
        avg_duration = sum(durations) / len(durations)
        return len(input_text.split()) / avg_duration

    def measure_energy(self):
        return None  # Placeholder for actual energy measurement

    def system_metadata(self):
        return {
            'OS': platform.system(),
            'Processor': platform.processor(),
            'Machine': platform.machine()
        }

    def benchmark(self, input_text, best_score, best_scored_llm, metrics_to_run=None):
        if not metrics_to_run:
            metrics_to_run = ['Memory', 'Throughput', 'Energy']

        results = {
            'Backend': self.backend,
            'Dtype': self.dtype,
            'Optimizations': self.llm.optimizations,
            'Quantization': self.llm.quantization,
            'Class': self.llm.class_name,
            'Type': self.llm.type_name,
            'System': self.system_metadata()
        }

        if 'Memory' in metrics_to_run:
            results['Memory (MB)'] = self.measure_memory()
        if 'Throughput' in metrics_to_run:
            results['Throughput (tokens/s)'] = self.measure_throughput(input_text)
        if 'Energy' in metrics_to_run:
            results['Energy (tokens/kWh)'] = self.measure_energy()
        
        results['Best Score (%)'] = best_score
        results['Best Scored LLM'] = best_scored_llm

        self.history.append(results)
        self.log_results(results)
        return results

    def log_results(self, results):
        print(colored("\n===== BENCHMARK RESULTS =====", 'blue'))
        for key, value in results.items():
            if key in ['Throughput (tokens/s)', 'Best Score (%)']:
                print(colored(f"{key}: {value}", 'green'))
            elif key in ['Memory (MB)', 'Energy (tokens/kWh)']:
                print(colored(f"{key}: {value}", 'red'))
            else:
                print(f"{key}: {value}")
        print(colored("=============================\n", 'blue'))

# llm_instance = LLM(backend="CPU", dtype="FP32", optimizations="Layer Norm", quantization="8-bit", class_name="Transformer", type_name="BERT")
# benchmarker = BenchmarkSuite(llm_instance)
# results = benchmarker.benchmark(input_text="This is a sample input text.", best_score=98.5, best_scored_llm="GPT-3", metrics_to_run=['Memory', 'Throughput'])
