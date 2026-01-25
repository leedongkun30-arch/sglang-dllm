# bench_creditdecoding.py
import time
import torch
from test_creditdecoding_pytest import (
    DummyModelRunner,
    DummyForwardBatch,
    run_original_like,
    run_optimized_like,
)

def benchmark(device="cuda", iters=50):
    B, T, K, V = 16, 64, 8, 4096   # 현실적인 크기
    mask_id = 999
    threshold = 0.95
    gamma, lam, alpha, eps = 0.65, 0.70, 0.50, 1e-6

    device = torch.device(device)
    runner = DummyModelRunner(vocab_size=V, device=device)

    input_ids = torch.randint(0, V, (B, T), device=device, dtype=torch.long)
    input_ids[:, -T//2:] = mask_id
    input_bt = input_ids.reshape(B * T).contiguous()

    fb1 = DummyForwardBatch(input_ids=input_bt.clone(), batch_size=B)
    fb2 = DummyForwardBatch(input_ids=input_bt.clone(), batch_size=B)

    # warmup
    for _ in range(10):
        run_optimized_like(runner, fb2, mask_id, T, K, threshold, gamma, lam, alpha, eps)

    torch.cuda.synchronize()

    # -------- torch baseline --------
    t0 = time.time()
    for _ in range(iters):
        run_original_like(runner, fb1, mask_id, T, K, threshold, gamma, lam, alpha, eps)
    torch.cuda.synchronize()
    t1 = time.time()

    # -------- triton optimized --------
    t2 = time.time()
    for _ in range(iters):
        run_optimized_like(runner, fb2, mask_id, T, K, threshold, gamma, lam, alpha, eps)
    torch.cuda.synchronize()
    t3 = time.time()

    print(f"\nDevice: {device}")
    print(f"Baseline (torch):  {(t1 - t0)/iters*1000:.2f} ms/iter")
    print(f"Optimized(triton): {(t3 - t2)/iters*1000:.2f} ms/iter")
    print(f"Speedup: {(t1 - t0)/(t3 - t2):.2f}x")


if __name__ == "__main__":
    benchmark("cuda", iters=50)
