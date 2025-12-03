import ray
import torch
import os
import time
import numpy as np

import ray.util.collective as collective


@ray.remote(num_gpus=8)
class Worker:
    def __init__(self):
        # Increased tensor size for more meaningful performance measurement.
        tensor_size = 1024 * 1024
        self.send_tensors = []
        for i in range(8):
            self.send_tensors.append(torch.ones((tensor_size,), dtype=torch.float32, device=f'cuda:{i}'))

        # Calculate the total size of tensors for this worker in bytes.
        self.tensor_bytes = sum(t.element_size() * t.nelement() for t in self.send_tensors)

    def setup(self, world_size, rank):
        self.world_size = world_size
        self.rank = rank
        collective.init_collective_group(world_size, rank, "nccl", "177")
        return True

    def compute(self, num_iters=20):
        # Warm-up iterations
        for _ in range(5):
            collective.allreduce_multigpu(self.send_tensors, "177")
            # Synchronize to wait for the operation to complete
            for i in range(8):
                torch.cuda.synchronize(device=f'cuda:{i}')

        # Timed iterations
        start_time = time.perf_counter()
        for _ in range(num_iters):
            collective.allreduce_multigpu(self.send_tensors, "177")
        
        # Synchronize to wait for all operations to complete
        for i in range(8):
            torch.cuda.synchronize(device=f'cuda:{i}')
        
        end_time = time.perf_counter()

        duration = end_time - start_time
        avg_time_per_iter = duration / num_iters

        # The amount of data communicated by a single node in a ring all-reduce is 2 * (N-1)/N * message_size.
        # Bandwidth is calculated based on this.
        # For multi-gpu allreduce within a node, the communication pattern is more complex.
        # A common way to report "bus bandwidth" is to consider the total size of data
        # on all GPUs.
        # Total data processed per iteration is tensor_bytes.
        # The algorithmic data transfer size for ring all-reduce is 2 * (world_size - 1) / world_size * tensor_bytes
        # We will use a simpler model: total bytes / time.
        # Effective bandwidth:
        # Each GPU sends and receives data. The total data moved is complex.
        # A common bus bandwidth calculation is based on the data size per GPU.
        # Let's use the simple model: Total Size / Time
        
        # The total size of data across all GPUs on this worker
        total_size_gb = self.tensor_bytes / 1e9
        
        # Bandwidth in GB/s
        bandwidth_gbs = total_size_gb / avg_time_per_iter

        # Algorithmic bandwidth for ring all-reduce.
        # This is a more accurate measure of the interconnect bandwidth utilization.
        algo_bandwidth_gbs = (2 * (self.world_size - 1) / self.world_size) * total_size_gb / avg_time_per_iter


        # The result of allreduce should be world_size
        # Verify one tensor
        expected_value = self.world_size
        is_correct = torch.allclose(self.send_tensors[0][0], torch.tensor(float(expected_value), device=self.send_tensors[0].device, dtype=self.send_tensors[0].dtype))

        return {
            "rank": self.rank,
            "avg_time_s": avg_time_per_iter,
            "total_size_gb": total_size_gb,
            "bandwidth_gbs": bandwidth_gbs,
            "algo_bandwidth_gbs": algo_bandwidth_gbs,
            "correctness_check": bool(is_correct)
        }


    def destroy(self):
        collective.destroy_collective_group("177")


if __name__ == "__main__":
    ray.init(address="auto")

    num_workers = 2
    workers = []
    init_rets = []

    for i in range(num_workers):
        w = Worker.remote()
        workers.append(w)
        init_rets.append(w.setup.remote(num_workers, i))
    
    ray.get(init_rets)
    print("Collective groups initialized.")

    results = ray.get([w.compute.remote() for w in workers])
    
    print("\n--- Allreduce Performance ---")
    for res in results:
        print(f"Worker {res['rank']}:")
        print(f"  Average time per iteration: {res['avg_time_s']:.6f} s")
        print(f"  Total data size per worker: {res['total_size_gb'] * 1000:.2f} MB")
        print(f"  Bandwidth (Total Size / Time): {res['bandwidth_gbs']:.2f} GB/s")
        print(f"  Algorithmic Ring Bandwidth: {res['algo_bandwidth_gbs']:.2f} GB/s")
        print(f"  Correctness check passed: {res['correctness_check']}")


    ray.get([w.destroy.remote() for w in workers])
    print("\nCollective groups destroyed.")

    ray.shutdown()