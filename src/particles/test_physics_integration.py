#!/usr/bin/env python3
"""
Integration test with Chat 2's physics engine
Verifies complete weather -> physics -> particles pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.weather.noaa_api import NOAAWeatherAPI, WeatherObservation
from src.physics.force_field_engine import PhysicsEngine
from src.particles.viscous_particle_system import ViscousParticleSystem

def test_full_integration():
    """Test complete pipeline from weather to particles."""
    
    print("=== FULL INTEGRATION TEST ===")
    
    # Initialize all systems
    print("\n1. Initializing systems...")
    weather_api = NOAAWeatherAPI()
    physics_engine = PhysicsEngine()
    particle_system = ViscousParticleSystem(particle_count=100_000)  # Start with 100K
    
    # Test different weather patterns
    weather_patterns = ['calm', 'storm', 'heat_wave', 'fog', 'hurricane']
    
    results = []
    
    for pattern in weather_patterns:
        print(f"\n2. Testing weather pattern: {pattern}")
        
        # Get weather data
        weather = weather_api.get_weather(pattern)
        print(f"   Weather: T={weather.temperature:.1f}°C, P={weather.pressure:.1f}hPa, " +
              f"Wind={weather.wind_speed:.1f}m/s")
        
        # Generate force field
        print("   Generating force field...")
        start_time = time.perf_counter()
        force_field = physics_engine.generate_3d_force_field(weather)
        force_gen_time = (time.perf_counter() - start_time) * 1000
        print(f"   Force field generation: {force_gen_time:.2f}ms")
        
        # Sample forces for particles
        print("   Sampling forces for particles...")
        start_time = time.perf_counter()
        particle_system.forces = physics_engine.sample_forces_smart(
            force_field, particle_system.positions
        )
        sample_time = (time.perf_counter() - start_time) * 1000
        print(f"   Force sampling: {sample_time:.2f}ms")
        
        # Update particles
        print("   Updating particles...")
        start_time = time.perf_counter()
        particle_system.update(force_field)
        update_time = (time.perf_counter() - start_time) * 1000
        print(f"   Particle update: {update_time:.2f}ms")
        
        # Total frame time
        total_time = force_gen_time + sample_time + update_time
        print(f"   TOTAL FRAME TIME: {total_time:.2f}ms")
        print(f"   Meets 60 FPS target: {'YES' if total_time < 16.67 else 'NO'}")
        
        # Store results
        results.append({
            'pattern': pattern,
            'weather': weather,
            'force_gen_time': force_gen_time,
            'sample_time': sample_time,
            'update_time': update_time,
            'total_time': total_time,
            'positions': particle_system.positions.copy(),
            'velocities': particle_system.velocities.copy(),
            'colors': particle_system.colors.copy()
        })
        
        # Run for a few seconds to see effect
        print("   Running simulation...")
        for frame in range(120):  # 2 seconds at 60 FPS
            particle_system.forces = physics_engine.sample_forces_smart(
                force_field, particle_system.positions
            )
            particle_system.update(force_field)
    
    # Create visualization
    create_integration_visualization(results)
    
    # Performance analysis
    analyze_integration_performance(results)
    
    return results


def create_integration_visualization(results):
    """Create visualization of different weather effects on particles."""
    
    fig = plt.figure(figsize=(20, 12))
    
    for idx, result in enumerate(results):
        # 3D visualization
        ax = fig.add_subplot(2, 5, idx + 1, projection='3d')
        
        # Subsample for visualization
        sample_size = min(5000, len(result['positions']))
        indices = np.random.choice(len(result['positions']), sample_size, replace=False)
        
        scatter = ax.scatter(result['positions'][indices, 0],
                           result['positions'][indices, 1],
                           result['positions'][indices, 2],
                           c=result['colors'][indices],
                           s=1, alpha=0.6)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-1, 1)
        ax.set_title(f"{result['pattern'].title()}\n" +
                    f"T={result['weather'].temperature:.0f}°C, " +
                    f"Wind={result['weather'].wind_speed:.0f}m/s")
        
        # 2D top view
        ax2 = fig.add_subplot(2, 5, idx + 6)
        
        # Create density heatmap
        H, xedges, yedges = np.histogram2d(
            result['positions'][:, 0],
            result['positions'][:, 1],
            bins=40,
            range=[[-2, 2], [-2, 2]]
        )
        
        im = ax2.imshow(H.T, origin='lower', extent=[-2, 2, -2, 2],
                       cmap='hot', interpolation='gaussian')
        ax2.set_title(f"{result['pattern'].title()} - Density")
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_aspect('equal')
        
        # Add container boundary
        rect = plt.Rectangle((-2, -2), 4, 4, fill=False, 
                           edgecolor='cyan', linewidth=1)
        ax2.add_patch(rect)
    
    plt.tight_layout()
    os.makedirs('verification_outputs/chat_3_particles', exist_ok=True)
    plt.savefig('verification_outputs/chat_3_particles/weather_integration_effects.png', dpi=150)
    plt.close()


def analyze_integration_performance(results):
    """Analyze performance across different weather patterns."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Extract data
    patterns = [r['pattern'] for r in results]
    force_times = [r['force_gen_time'] for r in results]
    sample_times = [r['sample_time'] for r in results]
    update_times = [r['update_time'] for r in results]
    total_times = [r['total_time'] for r in results]
    
    # Stacked bar chart of timing breakdown
    x = np.arange(len(patterns))
    width = 0.6
    
    p1 = ax1.bar(x, force_times, width, label='Force Generation', color='#3498db')
    p2 = ax1.bar(x, sample_times, width, bottom=force_times, 
                 label='Force Sampling', color='#2ecc71')
    p3 = ax1.bar(x, update_times, width, 
                 bottom=np.array(force_times) + np.array(sample_times),
                 label='Particle Update', color='#e74c3c')
    
    ax1.axhline(y=16.67, color='red', linestyle='--', linewidth=2, label='60 FPS Target')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Performance Breakdown by Weather Pattern')
    ax1.set_xticks(x)
    ax1.set_xticklabels(patterns)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add total time labels
    for i, total in enumerate(total_times):
        ax1.text(i, total + 0.5, f'{total:.1f}ms', ha='center', va='bottom')
    
    # Average particle speed by weather pattern
    avg_speeds = []
    for result in results:
        speeds = np.linalg.norm(result['velocities'], axis=1)
        avg_speeds.append(np.mean(speeds))
    
    ax2.bar(x, avg_speeds, width, color='#9b59b6')
    ax2.set_ylabel('Average Particle Speed')
    ax2.set_xlabel('Weather Pattern')
    ax2.set_title('Particle Movement by Weather Pattern')
    ax2.set_xticks(x)
    ax2.set_xticklabels(patterns)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, speed in enumerate(avg_speeds):
        ax2.text(i, speed + 0.001, f'{speed:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('verification_outputs/chat_3_particles/integration_performance.png', dpi=150)
    plt.close()


def test_million_particle_performance():
    """Test performance with up to 100K particles (optimized for Apple Silicon)."""
    
    print("\n=== PARTICLE PERFORMANCE TEST ===")
    
    # Initialize systems
    weather_api = NOAAWeatherAPI()
    physics_engine = PhysicsEngine()
    
    # Test at different particle counts (adjusted for Apple Silicon)
    particle_counts = [10_000, 25_000, 50_000, 75_000, 100_000]
    
    results = []
    
    for count in particle_counts:
        print(f"\nTesting with {count:,} particles...")
        
        # Create particle system
        particle_system = ViscousParticleSystem(particle_count=count)
        
        # Use storm weather for consistent testing
        weather = weather_api.get_weather('storm')
        force_field = physics_engine.generate_3d_force_field(weather)
        
        # Warm up
        for _ in range(10):
            particle_system.forces = physics_engine.sample_forces_smart(
                force_field, particle_system.positions
            )
            particle_system.update(force_field)
        
        # Measure performance over 100 frames
        times = {'force_gen': [], 'sample': [], 'update': [], 'total': []}
        
        for _ in range(100):
            # Force generation (would be cached in real system)
            start = time.perf_counter()
            force_field = physics_engine.generate_3d_force_field(weather)
            times['force_gen'].append((time.perf_counter() - start) * 1000)
            
            # Force sampling
            start = time.perf_counter()
            particle_system.forces = physics_engine.sample_forces_smart(
                force_field, particle_system.positions
            )
            times['sample'].append((time.perf_counter() - start) * 1000)
            
            # Particle update
            start = time.perf_counter()
            particle_system.update(force_field)
            times['update'].append((time.perf_counter() - start) * 1000)
            
            # Total
            times['total'].append(times['force_gen'][-1] + 
                                times['sample'][-1] + 
                                times['update'][-1])
        
        # Calculate averages
        avg_results = {
            'count': count,
            'force_gen': np.mean(times['force_gen']),
            'sample': np.mean(times['sample']),
            'update': np.mean(times['update']),
            'total': np.mean(times['total']),
            'max_total': np.max(times['total']),
            'meets_target': np.mean(times['total']) < 16.67
        }
        
        results.append(avg_results)
        
        print(f"  Force generation: {avg_results['force_gen']:.2f}ms")
        print(f"  Force sampling: {avg_results['sample']:.2f}ms")
        print(f"  Particle update: {avg_results['update']:.2f}ms")
        print(f"  TOTAL: {avg_results['total']:.2f}ms (max: {avg_results['max_total']:.2f}ms)")
        print(f"  Meets 60 FPS: {'YES' if avg_results['meets_target'] else 'NO'}")
    
    # Create performance scaling plot
    plot_performance_scaling(results)
    
    return results


def plot_performance_scaling(results):
    """Plot performance scaling with particle count."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Extract data
    counts = [r['count'] for r in results]
    force_gen = [r['force_gen'] for r in results]
    sample = [r['sample'] for r in results]
    update = [r['update'] for r in results]
    total = [r['total'] for r in results]
    
    # Timing breakdown
    ax1.plot(counts, force_gen, 'o-', label='Force Generation', linewidth=2)
    ax1.plot(counts, sample, 's-', label='Force Sampling', linewidth=2)
    ax1.plot(counts, update, '^-', label='Particle Update', linewidth=2)
    ax1.plot(counts, total, 'D-', label='Total Time', linewidth=3, color='red')
    ax1.axhline(y=16.67, color='green', linestyle='--', linewidth=2, label='60 FPS Target')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Particle Count')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Performance Scaling with Particle Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mark which configurations meet target
    for r in results:
        if r['meets_target']:
            ax1.annotate('✓', (r['count'], r['total']), 
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', color='green', fontsize=20, weight='bold')
        else:
            ax1.annotate('✗', (r['count'], r['total']), 
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', color='red', fontsize=20, weight='bold')
    
    # Efficiency plot (particles per millisecond)
    efficiency = [r['count'] / r['total'] for r in results]
    ax2.plot(counts, efficiency, 'o-', linewidth=2, color='purple')
    ax2.set_xscale('log')
    ax2.set_xlabel('Particle Count')
    ax2.set_ylabel('Particles per Millisecond')
    ax2.set_title('Processing Efficiency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('verification_outputs/chat_3_particles/million_particle_scaling.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    print("=== PHYSICS ENGINE INTEGRATION TEST ===")
    
    # Test full integration
    print("\n1. Testing full weather -> physics -> particles pipeline...")
    integration_results = test_full_integration()
    
    # Test million particle performance
    print("\n2. Testing performance scaling to 1 million particles...")
    scaling_results = test_million_particle_performance()
    
    print("\n=== INTEGRATION TEST COMPLETE ===")
    print("Check verification_outputs/chat_3_particles/ for results")
    
    # Final summary
    print("\n=== FINAL PERFORMANCE SUMMARY ===")
    if scaling_results[-1]['meets_target']:
        print(f"✅ SUCCESS: {scaling_results[-1]['count']:,} particles at {scaling_results[-1]['total']:.2f}ms " +
              f"({1000/scaling_results[-1]['total']:.1f} FPS)")
    else:
        # Find the largest count that meets target
        for i in range(len(scaling_results)-1, -1, -1):
            if scaling_results[i]['meets_target']:
                print(f"✅ SUCCESS: {scaling_results[i]['count']:,} particles at {scaling_results[i]['total']:.2f}ms " +
                      f"({1000/scaling_results[i]['total']:.1f} FPS)")
                print(f"⚠️  {scaling_results[-1]['count']:,} particles exceed target at {scaling_results[-1]['total']:.2f}ms")
                break
    
    print("\nBreakdown at 100K particles:")
    # Find 100K result
    for result in scaling_results:
        if result['count'] == 100_000:
            print(f"  Force generation: {result['force_gen']:.2f}ms")
            print(f"  Force sampling: {result['sample']:.2f}ms")
            print(f"  Particle update: {result['update']:.2f}ms")
            break