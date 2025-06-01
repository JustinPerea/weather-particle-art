#!/usr/bin/env python3
"""
Test batch sampling performance for force field
Verifies optimization for 1M particles at 60 FPS
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.weather.noaa_api import NOAAWeatherAPI
from src.physics.force_field_engine import PhysicsEngine

def test_batch_sampling_performance():
    """Test the optimized batch sampling performance"""
    print("=== BATCH SAMPLING PERFORMANCE TEST ===\n")
    
    # Initialize components
    weather_api = NOAAWeatherAPI()
    physics = PhysicsEngine()
    
    # Get a test weather pattern
    weather = weather_api.get_weather('storm')
    
    # Generate force field
    print("Generating force field...")
    start = time.time()
    force_field = physics.generate_3d_force_field(weather)
    generation_time = time.time() - start
    print(f"Force field generated in {generation_time*1000:.1f}ms\n")
    
    # Test different particle counts
    particle_counts = [1000, 10_000, 100_000, 1_000_000]
    
    print("Testing batch sampling performance:")
    print("-" * 60)
    print(f"{'Particles':>12} | {'Time (ms)':>12} | {'Per Particle':>12} | {'FPS':>8}")
    print("-" * 60)
    
    for count in particle_counts:
        # Generate random particle positions
        positions = np.random.rand(count, 3).astype(np.float32) * physics.box_size
        
        # Warm up
        _ = physics.sample_forces_batch(force_field, positions[:100])
        
        # Time the batch sampling
        start = time.time()
        forces = physics.sample_forces_batch(force_field, positions)
        batch_time = time.time() - start
        
        # Calculate metrics
        time_ms = batch_time * 1000
        time_per_particle = batch_time / count * 1e6  # microseconds
        max_fps = 1.0 / batch_time if batch_time > 0 else float('inf')
        
        # Verify output shape
        assert forces.shape == (count, 3), f"Wrong output shape: {forces.shape}"
        
        print(f"{count:>12,} | {time_ms:>12.2f} | {time_per_particle:>10.3f}μs | {max_fps:>8.1f}")
    
    print("-" * 60)
    
    # Test if we meet 60 FPS requirement for 1M particles
    million_time = batch_time  # Last test was 1M particles
    
    print(f"\n{'='*60}")
    print("PERFORMANCE ANALYSIS FOR 1M PARTICLES:")
    print(f"{'='*60}")
    print(f"Batch sampling time: {million_time*1000:.2f}ms")
    print(f"Frame budget (60 FPS): 16.67ms")
    print(f"Sampling uses: {million_time*1000/16.67*100:.1f}% of frame budget")
    
    if million_time * 1000 < 16.67:
        print("\n✅ PERFORMANCE TARGET MET! Can achieve 60 FPS with 1M particles")
        speedup = 13000 / (million_time * 1000)  # 13s old time vs new time
        print(f"Speedup: {speedup:.1f}x faster than single-particle sampling")
    else:
        print("\n❌ Performance needs further optimization")
        required_speedup = million_time * 1000 / 16.67
        print(f"Need {required_speedup:.1f}x more speed for 60 FPS")
    
    # Compare single vs batch sampling
    print(f"\n{'='*60}")
    print("SINGLE vs BATCH COMPARISON:")
    print(f"{'='*60}")
    
    # Time single particle sampling
    test_positions = positions[:1000]  # Test with 1000 particles
    
    # Single particle method (old way)
    start = time.time()
    single_forces = []
    for pos in test_positions:
        force = physics.sample_force_at_position(force_field, pos)
        single_forces.append(force)
    single_time = time.time() - start
    
    # Batch method (new way)
    start = time.time()
    batch_forces = physics.sample_forces_batch(force_field, test_positions)
    batch_time = time.time() - start
    
    print(f"1000 particles:")
    print(f"  Single method: {single_time*1000:.2f}ms ({single_time/1000*1e6:.1f}μs per particle)")
    print(f"  Batch method:  {batch_time*1000:.2f}ms ({batch_time/1000*1e6:.1f}μs per particle)")
    print(f"  Speedup: {single_time/batch_time:.1f}x")
    
    # Verify results match
    single_forces = np.array(single_forces)
    max_diff = np.max(np.abs(single_forces - batch_forces))
    print(f"\nMax difference between methods: {max_diff:.6f}")
    if max_diff < 1e-5:
        print("✅ Results match between single and batch methods")
    else:
        print("⚠️  Results differ between methods")

if __name__ == "__main__":
    test_batch_sampling_performance()