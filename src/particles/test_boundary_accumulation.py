#!/usr/bin/env python3
"""
Boundary Accumulation and Viscous Flow Verification
Demonstrates paint-like accumulation at container edges
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.particles.viscous_particle_system import ViscousParticleSystem

def test_boundary_accumulation():
    """Test and visualize boundary accumulation behavior."""
    
    # Create system with fewer particles for clear visualization
    system = ViscousParticleSystem(particle_count=50_000)
    
    # Create figure for multi-panel visualization
    fig = plt.figure(figsize=(16, 10))
    
    # Define test scenarios
    test_scenarios = [
        {
            'name': 'Initial State',
            'frames': 0,
            'force': None
        },
        {
            'name': 'Gentle Drift Right',
            'frames': 120,  # 2 seconds
            'force': lambda: apply_directional_force(system, [0.3, 0.0, 0.0])
        },
        {
            'name': 'Strong Push to Corner',
            'frames': 180,  # 3 seconds
            'force': lambda: apply_directional_force(system, [0.5, 0.5, 0.0])
        },
        {
            'name': 'Gravity Fall',
            'frames': 240,  # 4 seconds
            'force': lambda: apply_directional_force(system, [0.0, 0.0, -0.8])
        }
    ]
    
    # Run scenarios and capture states
    states = []
    
    for scenario in test_scenarios:
        print(f"Running scenario: {scenario['name']}")
        
        if scenario['frames'] > 0:
            for frame in range(scenario['frames']):
                if scenario['force']:
                    scenario['force']()
                system.update(None)
                
                # Capture state every 30 frames (0.5 seconds)
                if frame % 30 == 0:
                    positions = system.positions.copy()
                    velocities = system.velocities.copy()
                    colors = system.colors.copy()
                    states.append({
                        'name': scenario['name'],
                        'frame': frame,
                        'positions': positions,
                        'velocities': velocities,
                        'colors': colors
                    })
        else:
            # Initial state
            states.append({
                'name': scenario['name'],
                'frame': 0,
                'positions': system.positions.copy(),
                'velocities': system.velocities.copy(),
                'colors': system.colors.copy()
            })
    
    # Create visualization panels
    plot_boundary_accumulation_results(states, system.container_bounds)
    
    # Create density heatmaps
    create_density_heatmaps(states, system.container_bounds)
    
    # Analyze accumulation patterns
    analyze_accumulation_patterns(states, system.container_bounds)


def apply_directional_force(system, direction):
    """Apply uniform directional force to all particles."""
    force = np.array(direction, dtype=np.float32)
    system.forces[:] = force


def plot_boundary_accumulation_results(states, container_bounds):
    """Create comprehensive visualization of boundary accumulation."""
    
    # Select key states to visualize
    key_states = [states[0], states[3], states[6], states[-1]]
    
    fig = plt.figure(figsize=(16, 12))
    
    for idx, state in enumerate(key_states):
        # 3D view
        ax1 = fig.add_subplot(4, 3, idx*3 + 1, projection='3d')
        
        # Subsample for visualization
        sample_size = min(5000, len(state['positions']))
        indices = np.random.choice(len(state['positions']), sample_size, replace=False)
        
        scatter = ax1.scatter(state['positions'][indices, 0],
                            state['positions'][indices, 1],
                            state['positions'][indices, 2],
                            c=state['colors'][indices],
                            s=2, alpha=0.6)
        
        ax1.set_xlim(-container_bounds[0], container_bounds[0])
        ax1.set_ylim(-container_bounds[1], container_bounds[1])
        ax1.set_zlim(-container_bounds[2], container_bounds[2])
        ax1.set_title(f"{state['name']} - 3D View")
        
        # XY top view (to see lateral accumulation)
        ax2 = fig.add_subplot(4, 3, idx*3 + 2)
        ax2.scatter(state['positions'][indices, 0],
                   state['positions'][indices, 1],
                   c=state['colors'][indices],
                   s=1, alpha=0.4)
        ax2.set_xlim(-container_bounds[0], container_bounds[0])
        ax2.set_ylim(-container_bounds[1], container_bounds[1])
        ax2.set_aspect('equal')
        ax2.set_title(f"{state['name']} - Top View")
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        
        # Add boundary box
        rect = plt.Rectangle((-container_bounds[0], -container_bounds[1]),
                           2*container_bounds[0], 2*container_bounds[1],
                           fill=False, edgecolor='red', linewidth=2)
        ax2.add_patch(rect)
        
        # XZ side view (to see vertical accumulation)
        ax3 = fig.add_subplot(4, 3, idx*3 + 3)
        ax3.scatter(state['positions'][indices, 0],
                   state['positions'][indices, 2],
                   c=state['colors'][indices],
                   s=1, alpha=0.4)
        ax3.set_xlim(-container_bounds[0], container_bounds[0])
        ax3.set_ylim(-container_bounds[2], container_bounds[2])
        ax3.set_aspect('equal')
        ax3.set_title(f"{state['name']} - Side View")
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        
        # Add boundary box
        rect = plt.Rectangle((-container_bounds[0], -container_bounds[2]),
                           2*container_bounds[0], 2*container_bounds[2],
                           fill=False, edgecolor='red', linewidth=2)
        ax3.add_patch(rect)
    
    plt.tight_layout()
    os.makedirs('verification_outputs/chat_3_particles', exist_ok=True)
    plt.savefig('verification_outputs/chat_3_particles/boundary_accumulation_multi_view.png', dpi=150)
    plt.close()


def create_density_heatmaps(states, container_bounds):
    """Create density heatmaps showing accumulation patterns."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Select key states
    key_states = [states[0], states[3], states[6], states[-1]]
    
    for idx, state in enumerate(key_states):
        ax = axes[idx]
        
        # Create 2D histogram for XY plane
        H, xedges, yedges = np.histogram2d(
            state['positions'][:, 0],
            state['positions'][:, 1],
            bins=50,
            range=[[-container_bounds[0], container_bounds[0]],
                   [-container_bounds[1], container_bounds[1]]]
        )
        
        # Plot heatmap
        im = ax.imshow(H.T, origin='lower', 
                      extent=[-container_bounds[0], container_bounds[0],
                              -container_bounds[1], container_bounds[1]],
                      cmap='hot', interpolation='gaussian')
        
        ax.set_title(f"{state['name']} - Density Map")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Particle Density')
        
        # Add boundary
        rect = plt.Rectangle((-container_bounds[0], -container_bounds[1]),
                           2*container_bounds[0], 2*container_bounds[1],
                           fill=False, edgecolor='cyan', linewidth=2)
        ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig('verification_outputs/chat_3_particles/density_heatmaps.png', dpi=150)
    plt.close()


def analyze_accumulation_patterns(states, container_bounds):
    """Analyze and plot accumulation metrics."""
    
    # Define boundary zones (within 10% of container edge)
    boundary_threshold = 0.1
    
    accumulation_data = []
    
    for state in states:
        positions = state['positions']
        
        # Count particles near each boundary
        near_x_min = np.sum(positions[:, 0] < -container_bounds[0] * (1 - boundary_threshold))
        near_x_max = np.sum(positions[:, 0] > container_bounds[0] * (1 - boundary_threshold))
        near_y_min = np.sum(positions[:, 1] < -container_bounds[1] * (1 - boundary_threshold))
        near_y_max = np.sum(positions[:, 1] > container_bounds[1] * (1 - boundary_threshold))
        near_z_min = np.sum(positions[:, 2] < -container_bounds[2] * (1 - boundary_threshold))
        near_z_max = np.sum(positions[:, 2] > container_bounds[2] * (1 - boundary_threshold))
        
        total_near_boundary = (near_x_min + near_x_max + near_y_min + 
                              near_y_max + near_z_min + near_z_max)
        
        accumulation_data.append({
            'name': state['name'],
            'frame': state['frame'],
            'near_x_min': near_x_min,
            'near_x_max': near_x_max,
            'near_y_min': near_y_min,
            'near_y_max': near_y_max,
            'near_z_min': near_z_min,
            'near_z_max': near_z_max,
            'total_near_boundary': total_near_boundary,
            'percentage_at_boundary': (total_near_boundary / len(positions)) * 100
        })
    
    # Plot accumulation over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Extract data for plotting
    frames = [d['frame'] for d in accumulation_data]
    percentages = [d['percentage_at_boundary'] for d in accumulation_data]
    
    # Percentage at boundaries over time
    ax1.plot(frames, percentages, 'b-', linewidth=2)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Percentage of Particles at Boundaries (%)')
    ax1.set_title('Boundary Accumulation Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Add scenario labels
    current_scenario = None
    for i, data in enumerate(accumulation_data):
        if data['name'] != current_scenario:
            current_scenario = data['name']
            ax1.axvline(x=data['frame'], color='gray', linestyle='--', alpha=0.5)
            ax1.text(data['frame'], ax1.get_ylim()[1] * 0.9, 
                    current_scenario, rotation=45, ha='right')
    
    # Boundary distribution for final state
    final_state = accumulation_data[-1]
    boundaries = ['X-', 'X+', 'Y-', 'Y+', 'Z-', 'Z+']
    counts = [final_state['near_x_min'], final_state['near_x_max'],
             final_state['near_y_min'], final_state['near_y_max'],
             final_state['near_z_min'], final_state['near_z_max']]
    
    bars = ax2.bar(boundaries, counts, color=['red', 'darkred', 'green', 'darkgreen', 'blue', 'darkblue'])
    ax2.set_xlabel('Boundary')
    ax2.set_ylabel('Particle Count')
    ax2.set_title('Final Boundary Distribution')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.annotate(f'{int(count)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('verification_outputs/chat_3_particles/accumulation_analysis.png', dpi=150)
    plt.close()
    
    # Print summary
    print("\n=== BOUNDARY ACCUMULATION ANALYSIS ===")
    print(f"Initial particles at boundaries: {accumulation_data[0]['percentage_at_boundary']:.1f}%")
    print(f"Final particles at boundaries: {accumulation_data[-1]['percentage_at_boundary']:.1f}%")
    print(f"Maximum accumulation reached: {max(d['percentage_at_boundary'] for d in accumulation_data):.1f}%")
    
    print("\nFinal boundary distribution:")
    for boundary, count in zip(boundaries, counts):
        print(f"  {boundary}: {count} particles ({count/len(states[-1]['positions'])*100:.1f}%)")


def test_viscous_flow_patterns():
    """Test different viscous flow patterns."""
    
    system = ViscousParticleSystem(particle_count=30_000)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Different force patterns to test viscous behavior
    patterns = [
        ('Initial', None, 0),
        ('Circular Flow', 'circular', 120),
        ('Convergence', 'converge', 120),
        ('Shear Flow', 'shear', 120),
        ('Turbulent', 'turbulent', 120),
        ('Relaxation', None, 180)  # Let forces dissipate
    ]
    
    for idx, (name, pattern, frames) in enumerate(patterns):
        print(f"Testing pattern: {name}")
        
        # Apply forces and simulate
        for frame in range(frames):
            if pattern == 'circular':
                apply_circular_forces(system)
            elif pattern == 'converge':
                apply_convergent_forces(system)
            elif pattern == 'shear':
                apply_shear_forces(system)
            elif pattern == 'turbulent':
                apply_turbulent_forces(system)
            else:
                system.forces[:] = 0  # No forces
            
            system.update(None)
        
        # Plot result
        ax = axes[idx]
        
        # Get particles in middle Z slice
        slice_mask = np.abs(system.positions[:, 2]) < 0.2
        slice_positions = system.positions[slice_mask]
        slice_velocities = system.velocities[slice_mask]
        slice_colors = system.colors[slice_mask]
        
        # Subsample for clarity
        if len(slice_positions) > 2000:
            indices = np.random.choice(len(slice_positions), 2000, replace=False)
            slice_positions = slice_positions[indices]
            slice_velocities = slice_velocities[indices]
            slice_colors = slice_colors[indices]
        
        # Plot particles
        scatter = ax.scatter(slice_positions[:, 0], slice_positions[:, 1],
                           c=slice_colors, s=3, alpha=0.6)
        
        # Add velocity streamlines for subset
        if len(slice_positions) > 100:
            stream_indices = np.random.choice(len(slice_positions), 100, replace=False)
            ax.quiver(slice_positions[stream_indices, 0],
                     slice_positions[stream_indices, 1],
                     slice_velocities[stream_indices, 0],
                     slice_velocities[stream_indices, 1],
                     alpha=0.3, scale=3, width=0.002)
        
        ax.set_xlim(-system.container_bounds[0], system.container_bounds[0])
        ax.set_ylim(-system.container_bounds[1], system.container_bounds[1])
        ax.set_aspect('equal')
        ax.set_title(name)
        
        # Add container
        rect = plt.Rectangle((-system.container_bounds[0], -system.container_bounds[1]),
                           2*system.container_bounds[0], 2*system.container_bounds[1],
                           fill=False, edgecolor='white', linewidth=1)
        ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig('verification_outputs/chat_3_particles/viscous_flow_patterns.png', 
                dpi=150, facecolor='black')
    plt.close()


def apply_circular_forces(system):
    """Apply forces for circular flow pattern."""
    for i in range(system.particle_count):
        x, y = system.positions[i, 0], system.positions[i, 1]
        r = np.sqrt(x*x + y*y)
        if r > 0.01:
            system.forces[i, 0] = -y / r * 0.3
            system.forces[i, 1] = x / r * 0.3
            system.forces[i, 2] = 0


def apply_convergent_forces(system):
    """Apply forces toward center."""
    center = np.array([0, 0, 0], dtype=np.float32)
    for i in range(system.particle_count):
        diff = center - system.positions[i]
        dist = np.linalg.norm(diff)
        if dist > 0.01:
            system.forces[i] = diff / dist * 0.2


def apply_shear_forces(system):
    """Apply shear flow forces."""
    for i in range(system.particle_count):
        y = system.positions[i, 1]
        # Opposite flows at top and bottom
        system.forces[i, 0] = 0.3 * np.sign(y)
        system.forces[i, 1] = 0
        system.forces[i, 2] = 0


def apply_turbulent_forces(system):
    """Apply pseudo-turbulent forces."""
    # Random forces with spatial correlation
    t = time.time() * 0.1
    for i in range(system.particle_count):
        x, y, z = system.positions[i]
        # Perlin-noise-like effect using sine waves
        fx = 0.2 * np.sin(x * 2 + t) * np.cos(y * 3 - t)
        fy = 0.2 * np.cos(x * 3 + t) * np.sin(y * 2 + t)
        fz = 0.1 * np.sin(x * 2 + y * 2 + t)
        system.forces[i] = [fx, fy, fz]


if __name__ == "__main__":
    print("=== BOUNDARY ACCUMULATION & VISCOUS FLOW VERIFICATION ===")
    
    # Test boundary accumulation
    print("\n1. Testing boundary accumulation behavior...")
    test_boundary_accumulation()
    
    # Test viscous flow patterns
    print("\n2. Testing viscous flow patterns...")
    test_viscous_flow_patterns()
    
    print("\n=== VERIFICATION COMPLETE ===")
    print("Check verification_outputs/chat_3_particles/ for results")