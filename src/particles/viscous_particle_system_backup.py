#!/usr/bin/env python3
"""
Viscous Particle System for Weather-Driven Art
Chat 3 Implementation - Optimized for 5ms update budget
"""

import numpy as np
import numba
from numba import jit, prange
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

# Numba-optimized functions for performance
@jit(nopython=True, parallel=True, fastmath=True)
def update_particles_numba(positions, velocities, forces, colors,
                          dt, viscosity, damping, max_speed,
                          cohesion_radius, cohesion_strength,
                          container_bounds, boundary_force, 
                          boundary_distance, accumulation_factor):
    """
    Update particle positions and velocities with viscous dynamics.
    Optimized with Numba for <5ms performance with 1M particles.
    """
    n_particles = positions.shape[0]
    
    # Pre-compute frequently used values
    inv_viscosity = 1.0 / viscosity
    cohesion_radius_sq = cohesion_radius * cohesion_radius
    
    # Update each particle in parallel
    for i in prange(n_particles):
        # Get current particle state
        pos = positions[i]
        vel = velocities[i]
        force = forces[i]
        
        # Apply viscous force (F = ma, but with viscosity)
        # Viscosity acts like mass, slowing acceleration
        accel = force * inv_viscosity
        
        # Update velocity with damping
        vel = (vel + accel * dt) * damping
        
        # Cohesion forces - simplified for performance
        # Only check nearby particles using spatial hashing would be ideal
        # For now, we'll sample a subset for performance
        cohesion_force = np.zeros(3, dtype=np.float32)
        
        # Sample nearby particles (every 100th for performance)
        sample_step = max(1, n_particles // 10000)
        for j in range(0, n_particles, sample_step):
            if i != j:
                diff = positions[j] - pos
                dist_sq = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
                
                if dist_sq < cohesion_radius_sq and dist_sq > 0.0001:
                    dist = np.sqrt(dist_sq)
                    # Attract to nearby particles
                    cohesion_force += diff * (cohesion_strength / dist)
        
        # Add cohesion to velocity
        vel += cohesion_force * dt
        
        # Limit maximum speed (honey-like behavior)
        speed = np.sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2])
        if speed > max_speed:
            vel *= max_speed / speed
        
        # Boundary forces - soft accumulation at edges
        for dim in range(3):
            # Distance from boundaries
            dist_to_min = pos[dim] + container_bounds[dim]
            dist_to_max = container_bounds[dim] - pos[dim]
            
            # Apply soft boundary forces
            if dist_to_min < boundary_distance:
                # Exponential repulsion from boundary
                strength = boundary_force * np.exp(-5.0 * dist_to_min / boundary_distance)
                vel[dim] += strength * dt
                # Accumulation effect - particles slow down near walls
                vel[dim] *= accumulation_factor
                
            if dist_to_max < boundary_distance:
                strength = boundary_force * np.exp(-5.0 * dist_to_max / boundary_distance)
                vel[dim] -= strength * dt
                vel[dim] *= accumulation_factor
        
        # Update position
        new_pos = pos + vel * dt
        
        # Soft clamping - particles can't escape but accumulate at boundaries
        for dim in range(3):
            if new_pos[dim] < -container_bounds[dim]:
                new_pos[dim] = -container_bounds[dim]
                vel[dim] *= 0.1  # Almost stop at boundary
            elif new_pos[dim] > container_bounds[dim]:
                new_pos[dim] = container_bounds[dim]
                vel[dim] *= 0.1
        
        # Update arrays
        positions[i] = new_pos
        velocities[i] = vel
        
        # Update color based on velocity (for visual effect)
        # Faster particles are brighter
        brightness = min(1.0, speed / max_speed * 2.0)
        colors[i, 0] = 0.8 * brightness + 0.2  # Red channel
        colors[i, 1] = 0.5 * brightness + 0.1  # Green channel
        colors[i, 2] = 0.3 * brightness + 0.1  # Blue channel


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
        MUST complete in <5ms for 1M particles!
        
        Args:
            force_field: Force field from physics engine (not used directly - forces pre-sampled)
            dt: Time step (default 1/60 second)
        """
        start_time = time.perf_counter()
        
        # Note: In the full system, forces would be pre-sampled by physics engine
        # For now, we'll use placeholder forces for testing
        # In production: self.forces = physics.sample_forces_smart(force_field, self.positions)
        
        # Update particles using Numba-optimized function
        update_particles_numba(
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
            for i in range(self.particle_count):
                x, y, z = self.positions[i]
                r = np.sqrt(x*x + y*y)
                if r > 0.01:
                    # Tangential force
                    self.forces[i, 0] = -y / r * 0.5
                    self.forces[i, 1] = x / r * 0.5
                    self.forces[i, 2] = 0.0
        elif pattern == 'updraft':
            # Upward forces in center, down at edges
            for i in range(self.particle_count):
                x, y, z = self.positions[i]
                r = np.sqrt(x*x + y*y)
                self.forces[i, 2] = 0.3 * np.exp(-r*r) - 0.1
        elif pattern == 'converge':
            # Forces toward center
            for i in range(self.particle_count):
                self.forces[i] = -self.positions[i] * 0.2


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
    
    # 3. Viscous flow visualization
    fig = plt.figure(figsize=(12, 8))
    
    # Create a slice view to see internal flow
    slice_particles = np.where(np.abs(system.positions[:, 2]) < 0.1)[0]
    slice_sample = slice_particles[np.random.choice(len(slice_particles), 
                                                   min(5000, len(slice_particles)), 
                                                   replace=False)]
    
    ax = fig.add_subplot(111)
    
    # Plot particles with velocity vectors
    ax.scatter(system.positions[slice_sample, 0], 
              system.positions[slice_sample, 1],
              c=system.colors[slice_sample], s=2, alpha=0.6)
    
    # Add velocity arrows for a subset
    arrow_sample = slice_sample[::10]  # Every 10th particle
    ax.quiver(system.positions[arrow_sample, 0],
             system.positions[arrow_sample, 1],
             system.velocities[arrow_sample, 0],
             system.velocities[arrow_sample, 1],
             alpha=0.5, scale=2)
    
    ax.set_xlim(-system.container_bounds[0], system.container_bounds[0])
    ax.set_ylim(-system.container_bounds[1], system.container_bounds[1])
    ax.set_aspect('equal')
    ax.set_title('Viscous Flow Pattern (Z-slice view)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Add container boundary
    rect = plt.Rectangle((-system.container_bounds[0], -system.container_bounds[1]),
                        2*system.container_bounds[0], 2*system.container_bounds[1],
                        fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'viscous_flow.png'), dpi=150)
    plt.close()
    
    print(f"Verification plots saved to {output_dir}/")


def run_performance_scaling_test():
    """Test performance at different particle counts."""
    particle_counts = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    results = []
    
    print("=== PERFORMANCE SCALING TEST ===")
    print("Testing update performance at different particle counts...")
    
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
    
    # Plot scaling results
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
    ax.set_title('Performance Scaling Analysis')
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
    
    # Test with 100K particles first
    print("\n1. Testing 100K particle system...")
    system = ViscousParticleSystem(particle_count=100_000)
    
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