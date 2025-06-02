#!/usr/bin/env python3
"""
Quick test to verify Numba is working correctly
"""

import numpy as np
import numba
from numba import jit
import time

# Test 1: Simple function
@jit(nopython=True)
def add_arrays_numba(a, b):
    return a + b

def add_arrays_python(a, b):
    return a + b

# Test 2: More complex function like our particle update
@jit(nopython=True, parallel=True)
def update_positions_numba(positions, velocities, dt):
    n = positions.shape[0]
    for i in numba.prange(n):
        positions[i] += velocities[i] * dt
    return positions

def update_positions_python(positions, velocities, dt):
    return positions + velocities * dt

def test_numba():
    print("Testing Numba performance...")
    print(f"Numba version: {numba.__version__}")
    
    # Create test data
    n = 100000
    positions = np.random.randn(n, 3).astype(np.float32)
    velocities = np.random.randn(n, 3).astype(np.float32)
    dt = 0.016
    
    # Warm up Numba (compile)
    print("\nWarming up Numba...")
    _ = update_positions_numba(positions.copy(), velocities, dt)
    
    # Test Python version
    print("\nTesting pure Python...")
    start = time.perf_counter()
    for _ in range(10):
        pos_py = update_positions_python(positions.copy(), velocities, dt)
    python_time = (time.perf_counter() - start) / 10 * 1000
    print(f"Python time: {python_time:.2f}ms")
    
    # Test Numba version
    print("\nTesting Numba...")
    start = time.perf_counter()
    for _ in range(10):
        pos_nb = update_positions_numba(positions.copy(), velocities, dt)
    numba_time = (time.perf_counter() - start) / 10 * 1000
    print(f"Numba time: {numba_time:.2f}ms")
    
    # Results
    print(f"\nSpeedup: {python_time/numba_time:.1f}x")
    
    if numba_time > python_time:
        print("\n❌ ERROR: Numba is SLOWER than Python!")
        print("This suggests Numba compilation is failing.")
        print("Possible issues:")
        print("- Numba not compatible with Python version")
        print("- CPU architecture issues")
        print("- Installation problems")
    elif numba_time < 1.0:
        print("\n✅ Numba is working correctly!")
    else:
        print("\n⚠️ Numba is working but slower than expected")

if __name__ == "__main__":
    test_numba()