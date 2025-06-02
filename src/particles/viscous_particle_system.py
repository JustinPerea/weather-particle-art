#!/usr/bin/env python3
"""
Viscous Particle System for Weather-Driven Art
Optimized version for Apple Silicon without Numba
"""

import numpy as np
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Particle behavior parameters matching Anadol aesthetic
@dataclass
class ParticleParams:
    viscosity: float = 100.0      # Extreme honey/lava flow
    damping: float = 0.95         # Strong resistance to movement
    max_speed: float = 0.1        # Very slow maximum speed
    cohesion_radius: float = 0.1  # Distance for particle clustering
    cohesion_strength: float = 0.3 # Strength of particle attraction
    boundary_force: float = 2.0    # Soft boundary repulsion strength
    boundary_distance: float = 0.2 # Distance from wall to start slowing
    accumulation_factor: float = 0.98 # How much particles slow at boundaries


def update_particles_vectorized(positions, velocities, forces, colors,
                               dt, viscosity, damping, max_speed,
                               cohesion_radius, cohesion_strength,
                               container_bounds, boundary_force, 
                               boundary_distance, accumulation_factor):
    """
    Update particle positions and velocities with viscous dynamics.
    Fully vectorized for performance without Numba.
    """
    # Apply viscous force (F = ma, but with viscosity)
    accel = forces / viscosity
    
    # Update velocity with damping
    velocities += accel * dt
    velocities *= damping
    
    # Simplified cohesion - sample nearby particles
    # For performance, we'll apply a global cohesion effect
    if cohesion_strength > 0:
        # Calculate center of mass
        center = np.mean(positions, axis=0)
        # Pull particles gently toward center
        cohesion_force = (center - positions) * cohesion_strength * 0.01
        velocities += cohesion_force * dt
    
    # Limit maximum speed (vectorized)
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    mask = speeds > max_speed
    velocities[mask.squeeze()] *= max_speed / speeds[mask.squeeze()]
    
    # Boundary forces - soft accumulation at edges
    for dim in range(3):
        # Distance from boundaries
        dist_to_min = positions[:, dim] + container_bounds[dim]
        dist_to_max = container_bounds[dim] - positions[:, dim]
        
        # Apply soft boundary forces where close to edges
        mask_min = dist_to_min < boundary_distance
        mask_max = dist_to_max < boundary_distance
        
        # Exponential repulsion
        strength_min = boundary_force * np.exp(-5.0 * dist_to_min[mask_min] / boundary_distance)
        strength_max = boundary_force * np.exp(-5.0 * dist_to_max[mask_max] / boundary_distance)
        
        velocities[mask_min, dim] += strength_min * dt
        velocities[mask_max, dim] -= strength_max * dt
        
        # Accumulation effect
        velocities[mask_min | mask_max, dim] *= accumulation_factor
    
    # Update positions
    positions += velocities * dt
    
    # Soft clamping - particles can't escape but accumulate at boundaries
    for dim in range(3):
        # Find particles outside bounds
        mask_min = positions[:, dim] < -container_bounds[dim]
        mask_max = positions[:, dim] > container_bounds[dim]
        
        # Clamp positions
        positions[mask_min, dim] = -container_bounds[dim]
        positions[mask_max, dim] = container_bounds[dim]
        
        # Almost stop at boundary
        velocities[mask_min | mask_max, dim] *= 0.1
    
    # Update colors based on velocity
    speeds_normalized = np.minimum(speeds / max_speed * 2.0, 1.0)
    colors[:, 0] = 0.8 * speeds_normalized.squeeze() + 0.2  # Red
    colors[:, 1] = 0.5 * speeds_normalized.squeeze() + 0.1  # Green
    colors[:, 2] = 0.3 * speeds_normalized.squeeze() + 0.1  # Blue
    
    return positions, velocities, colors


class ViscousParticleSystem:
    """High-performance viscous particle system for weather-driven art."""
    
    def __init__(self, particle_count=1_000_000, container_bounds=None):
        """
        Initialize particle system with pre-allocated arrays.
        
        Args:
            particle_count: Number of particles (default 1M)
            container_bounds: [x, y, z] half-extents of container (default [2, 2, 1])
        """
        self.particle_count = particle_count
        self.container_bounds = np.array(container_bounds or [2.0, 2.0, 1.0], dtype=np.float32)
        
        # Pre-allocate all arrays as float32 for memory efficiency
        self.positions = np.zeros((particle_count, 3), dtype=np.float32)
        self.velocities = np.zeros((particle_count, 3), dtype=np.float32)
        self.colors = np.ones((particle_count, 3), dtype=np.float32)
        
        # Temporary array for forces from physics engine
        self.forces = np.zeros((particle_count, 3), dtype=np.float32)
        
        # Particle parameters
        self.params = ParticleParams()
        
        # Initialize particles with interesting distribution
        self._initialize_particles()
        
        # Performance tracking
        self.update_times = []
        
    def _initialize_particles(self):
        """Initialize particles with clustered distribution for visual interest."""
        # Create several clusters for more interesting initial state
        n_clusters = 5
        particles_per_cluster = self.particle_count // n_clusters
        
        for i in range(n_clusters):
            start_idx = i * particles_per_cluster
            end_idx = start_idx + particles_per_cluster
            
            # Random cluster center within container
            center = np.random.uniform(-self.container_bounds * 0.5, 
                                     self.container_bounds * 0.5, 
                                     size=3).astype(np.float32)
            
            # Particles distributed around cluster center
            cluster_positions = np.random.normal(center, 0.2, 
                                               size=(particles_per_cluster, 3)).astype(np.float32)
            
            # Clamp to container
            for dim in range(3):
                cluster_positions[:, dim] = np.clip(cluster_positions[:, dim],
                                                   -self.container_bounds[dim] * 0.9,
                                                   self.container_bounds[dim] * 0.9)
            
            self.positions[start_idx:end_idx] = cluster_positions
            
            # Small initial velocities
            self.velocities[start_idx:end_idx] = np.random.normal(0, 0.01, 
                                                                 size=(particles_per_cluster, 3)).astype(np.float32)
        
        # Handle remaining particles
        remaining = self.particle_count - (n_clusters * particles_per_cluster)
        if remaining > 0:
            self.positions[-remaining:] = np.random.uniform(-self.container_bounds * 0.8,
                                                          self.container_bounds * 0.8,
                                                          size=(remaining, 3)).astype(np.float32)
    
    def update(self, force_field: np.ndarray, dt: float = 0.016) -> None:
        """
        Update all particles for one frame.
        
        Args:
            force_field: Force field from physics engine (not used directly)
            dt: Time step (default 1/60 second)
        """
        start_time = time.perf_counter()
        
        # Update particles using vectorized function
        self.positions, self.velocities, self.colors = update_particles_vectorized(
            self.positions, self.velocities, self.forces, self.colors,
            dt, self.params.viscosity, self.params.damping, self.params.max_speed,
            self.params.cohesion_radius, self.params.cohesion_strength,
            self.container_bounds, self.params.boundary_force,
            self.params.boundary_distance, self.params.accumulation_factor
        )
        
        # Track performance
        update_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        self.update_times.append(update_time)
        
        # Keep only recent history
        if len(self.update_times) > 100:
            self.update_times.pop(0)
    
    def get_render_data(self) -> tuple:
        """
        Get particle data for rendering.
        
        Returns:
            tuple: (positions, velocities, colors) arrays
        """
        return (self.positions, self.velocities, self.colors)
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        if not self.update_times:
            return {'avg_update_time': 0, 'max_update_time': 0, 'updates_over_5ms': 0}
        
        avg_time = np.mean(self.update_times)
        max_time = np.max(self.update_times)
        over_5ms = sum(1 for t in self.update_times if t > 5.0)
        
        return {
            'avg_update_time': avg_time,
            'max_update_time': max_time,
            'updates_over_5ms': over_5ms,
            'percentage_over_5ms': (over_5ms / len(self.update_times)) * 100
        }
    
    def apply_test_forces(self, pattern='swirl'):
        """Apply test force patterns for verification."""
        if pattern == 'swirl':
            # Circular swirling forces
            x = self.positions[:, 0]
            y = self.positions[:, 1]
            r = np.sqrt(x*x + y*y)
            mask = r > 0.01
            self.forces[mask, 0] = -y[mask] / r[mask] * 0.5
            self.forces[mask, 1] = x[mask] / r[mask] * 0.5
            self.forces[:, 2] = 0.0
        elif pattern == 'updraft':
            # Upward forces in center, down at edges
            x = self.positions[:, 0]
            y = self.positions[:, 1]
            r = np.sqrt(x*x + y*y)
            self.forces[:, 2] = 0.3 * np.exp(-r*r) - 0.1
        elif pattern == 'converge':
            # Forces toward center
            self.forces = -self.positions * 0.2


def create_verification_plots(system, output_dir='verification_outputs/chat_3_particles'):
    """Create comprehensive verification plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 3D particle distribution
    fig = plt.figure(figsize=(15, 5))
    
    # Subsample for visualization (can't plot 1M points effectively)
    sample_size = min(10000, system.particle_count)
    indices = np.random.choice(system.particle_count, sample_size, replace=False)
    
    # Initial distribution
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(system.positions[indices, 0], 
               system.positions[indices, 1], 
               system.positions[indices, 2],
               c=system.colors[indices], s=1, alpha=0.5)
    ax1.set_title('Initial Particle Distribution')
    ax1.set_xlim(-system.container_bounds[0], system.container_bounds[0])
    ax1.set_ylim(-system.container_bounds[1], system.container_bounds[1])
    ax1.set_zlim(-system.container_bounds[2], system.container_bounds[2])
    
    # Apply swirl pattern and update
    system.apply_test_forces('swirl')
    for _ in range(60):  # 1 second of simulation
        system.update(None)
    
    # After swirl
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(system.positions[indices, 0], 
               system.positions[indices, 1], 
               system.positions[indices, 2],
               c=system.colors[indices], s=1, alpha=0.5)
    ax2.set_title('After Swirl Forces (1s)')
    ax2.set_xlim(-system.container_bounds[0], system.container_bounds[0])
    ax2.set_ylim(-system.container_bounds[1], system.container_bounds[1])
    ax2.set_zlim(-system.container_bounds[2], system.container_bounds[2])
    
    # Boundary accumulation test
    system.apply_test_forces('converge')
    for _ in range(120):  # 2 more seconds
        system.update(None)
    
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(system.positions[indices, 0], 
               system.positions[indices, 1], 
               system.positions[indices, 2],
               c=system.colors[indices], s=1, alpha=0.5)
    ax3.set_title('Boundary Accumulation Test')
    ax3.set_xlim(-system.container_bounds[0], system.container_bounds[0])
    ax3.set_ylim(-system.container_bounds[1], system.container_bounds[1])
    ax3.set_zlim(-system.container_bounds[2], system.container_bounds[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'particle_distribution.png'), dpi=150)
    plt.close()
    
    # 2. Performance analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Update time history
    if system.update_times:
        ax1.plot(system.update_times, 'b-', alpha=0.7)
        ax1.axhline(y=5.0, color='r', linestyle='--', label='5ms target')
        ax1.set_ylabel('Update Time (ms)')
        ax1.set_xlabel('Frame')
        ax1.set_title(f'Update Performance ({system.particle_count:,} particles)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Performance statistics
        stats = system.get_performance_stats()
        stats_text = f"Average: {stats['avg_update_time']:.2f}ms\n"
        stats_text += f"Max: {stats['max_update_time']:.2f}ms\n"
        stats_text += f"Over 5ms: {stats['percentage_over_5ms']:.1f}%"
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Velocity distribution histogram
    speeds = np.linalg.norm(system.velocities, axis=1)
    ax2.hist(speeds, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(x=system.params.max_speed, color='r', linestyle='--', label=f'Max speed: {system.params.max_speed}')
    ax2.set_xlabel('Particle Speed')
    ax2.set_ylabel('Count')
    ax2.set_title('Velocity Distribution (Viscous Behavior Check)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_analysis.png'), dpi=150)
    plt.close()
    
    print(f"Verification plots saved to {output_dir}/")


def run_performance_scaling_test():
    """Test performance at different particle counts."""
    # Start with smaller counts for Apple Silicon
    particle_counts = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
    results = []
    
    print("=== PERFORMANCE SCALING TEST ===")
    print("Testing update performance at different particle counts...")
    print("(Optimized for Apple Silicon without Numba)")
    
    for count in particle_counts:
        print(f"\nTesting {count:,} particles...")
        system = ViscousParticleSystem(particle_count=count)
        
        # Warm up
        for _ in range(10):
            system.apply_test_forces('swirl')
            system.update(None)
        
        # Measure
        times = []
        for _ in range(100):
            system.apply_test_forces('swirl')
            start = time.perf_counter()
            system.update(None)
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        
        results.append({
            'count': count,
            'avg_time': avg_time,
            'max_time': max_time,
            'meets_target': avg_time < 5.0
        })
        
        print(f"  Average update time: {avg_time:.2f}ms")
        print(f"  Maximum update time: {max_time:.2f}ms")
        print(f"  Meets 5ms target: {'YES' if avg_time < 5.0 else 'NO'}")
    
    # Try larger counts if performance is good
    if results[-1]['meets_target']:
        print("\nTrying larger particle counts...")
        for count in [200_000, 500_000]:
            print(f"\nTesting {count:,} particles...")
            system = ViscousParticleSystem(particle_count=count)
            
            # Quick test
            times = []
            for _ in range(20):
                system.apply_test_forces('swirl')
                start = time.perf_counter()
                system.update(None)
                times.append((time.perf_counter() - start) * 1000)
            
            avg_time = np.mean(times)
            if avg_time > 20:  # Stop if too slow
                print(f"  Too slow: {avg_time:.2f}ms - skipping larger counts")
                break
                
            results.append({
                'count': count,
                'avg_time': avg_time,
                'max_time': np.max(times),
                'meets_target': avg_time < 5.0
            })
            
            print(f"  Average update time: {avg_time:.2f}ms")
    
    # Plot results
    plot_performance_scaling(results)
    
    return results


def plot_performance_scaling(results):
    """Plot performance scaling results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    counts = [r['count'] for r in results]
    avg_times = [r['avg_time'] for r in results]
    max_times = [r['max_time'] for r in results]
    
    ax.plot(counts, avg_times, 'b-o', label='Average time', linewidth=2)
    ax.plot(counts, max_times, 'r--o', label='Maximum time', linewidth=2)
    ax.axhline(y=5.0, color='green', linestyle='--', label='5ms target', linewidth=2)
    
    ax.set_xscale('log')
    ax.set_xlabel('Particle Count')
    ax.set_ylabel('Update Time (ms)')
    ax.set_title('Performance Scaling Analysis (Apple Silicon Optimized)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    for r in results:
        if r['meets_target']:
            ax.annotate('✓', (r['count'], r['avg_time']), 
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', color='green', fontsize=16, weight='bold')
    
    plt.tight_layout()
    os.makedirs('verification_outputs/chat_3_particles', exist_ok=True)
    plt.savefig('verification_outputs/chat_3_particles/performance_scaling.png', dpi=150)
    plt.close()
    
    return results


if __name__ == "__main__":
    print("=== VISCOUS PARTICLE SYSTEM VERIFICATION ===")
    print("Running Apple Silicon optimized version (without Numba)")
    
    # Test with smaller particle count first
    print("\n1. Testing 10K particle system...")
    system = ViscousParticleSystem(particle_count=10_000)
    
    # Run some updates to measure performance
    print("   Running performance test...")
    for i in range(100):
        system.apply_test_forces('swirl' if i < 50 else 'updraft')
        system.update(None)
    
    stats = system.get_performance_stats()
    print(f"   Average update time: {stats['avg_update_time']:.2f}ms")
    print(f"   Maximum update time: {stats['max_update_time']:.2f}ms")
    print(f"   Updates over 5ms: {stats['percentage_over_5ms']:.1f}%")
    
    # Create verification plots
    print("\n2. Creating verification plots...")
    create_verification_plots(system)
    
    # Run scaling test
    print("\n3. Running performance scaling test...")
    scaling_results = run_performance_scaling_test()
    
    print("\n=== VERIFICATION COMPLETE ===")
    print("Check verification_outputs/chat_3_particles/ for visual results")
    
    # Estimate max particles for 5ms target
    if scaling_results:
        for i in range(len(scaling_results) - 1):
            if scaling_results[i]['meets_target'] and not scaling_results[i+1]['meets_target']:
                max_particles = scaling_results[i]['count']
                print(f"\n💡 Maximum particles for 5ms target: ~{max_particles:,}")
                print(f"   Performance at {max_particles:,}: {scaling_results[i]['avg_time']:.2f}ms")
                break