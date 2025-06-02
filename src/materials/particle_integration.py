#!/usr/bin/env python3
"""
Integration helper to upgrade existing particle systems to geometry-based rendering
This ensures particles are visible with emission effects
"""

import bpy
import numpy as np


def upgrade_to_geometry_particles(renderer, particle_system, weather_data, max_visible=10000):
    """
    Upgrade an existing particle system to use geometry instances
    
    Args:
        renderer: The particle renderer object
        particle_system: The particle system with positions
        weather_data: Current weather observation
        max_visible: Maximum number of visible particles (for performance)
    """
    print("\n=== UPGRADING TO GEOMETRY PARTICLES ===")
    
    # Get particle positions
    if hasattr(particle_system, 'positions'):
        positions = particle_system.positions
    elif hasattr(particle_system, 'particles'):
        positions = particle_system.particles
    else:
        print("❌ Could not find particle positions")
        return None
    
    total_particles = len(positions)
    print(f"Total particles: {total_particles:,}")
    
    # Calculate step for subsampling if needed
    if total_particles > max_visible:
        step = total_particles // max_visible
        visible_positions = positions[::step]
        print(f"Using every {step}th particle: {len(visible_positions):,} visible")
    else:
        visible_positions = positions
        print(f"All {len(visible_positions):,} particles will be visible")
    
    # Create base particle if it doesn't exist
    base_particle = bpy.data.objects.get("BaseParticle")
    if not base_particle:
        # Calculate appropriate size
        if len(visible_positions) > 5000:
            radius = 0.005
        elif len(visible_positions) > 1000:
            radius = 0.01
        else:
            radius = 0.02
            
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=1,
            radius=radius,
            location=(0, 0, -10)
        )
        base_particle = bpy.context.active_object
        base_particle.name = "BaseParticle"
        print(f"✅ Created base particle (radius: {radius})")
    
    # Apply material if available
    material = None
    if hasattr(renderer, 'material_system') and hasattr(renderer.material_system, 'material'):
        material = renderer.material_system.material
    elif hasattr(renderer, 'material'):
        material = renderer.material
    
    if material and base_particle:
        base_particle.data.materials.clear()
        base_particle.data.materials.append(material)
        print(f"✅ Applied material: {material.name}")
    
    # Create collection for instances
    collection_name = "GeometryParticles"
    if collection_name in bpy.data.collections:
        particle_collection = bpy.data.collections[collection_name]
        # Clear existing
        for obj in particle_collection.objects:
            bpy.data.objects.remove(obj)
    else:
        particle_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(particle_collection)
    
    # Create instances
    print(f"Creating {len(visible_positions):,} particle instances...")
    for i, pos in enumerate(visible_positions):
        instance = base_particle.copy()
        instance.data = base_particle.data  # Share mesh
        instance.location = pos
        particle_collection.objects.link(instance)
        
        if i % 1000 == 0 and i > 0:
            print(f"  Progress: {i:,}/{len(visible_positions):,}")
    
    print("✅ Geometry particles created!")
    
    # Hide original vertex-based particles if they exist
    if hasattr(renderer, 'obj'):
        renderer.obj.hide_viewport = True
        renderer.obj.hide_render = True
        print("✅ Hidden original vertex particles")
    
    # Update viewport
    bpy.context.view_layer.update()
    
    return particle_collection


def create_geometry_node_setup(container_object, base_particle, particle_count):
    """
    Create an efficient geometry node setup for large particle counts
    This is more performant than individual instances
    """
    print("\n=== CREATING GEOMETRY NODE PARTICLE SYSTEM ===")
    
    # Add geometry nodes modifier
    geo_mod = container_object.modifiers.new("ParticleInstancer", 'NODES')
    
    # Create node group
    node_group = bpy.data.node_groups.new("ParticleInstancer", 'GeometryNodeTree')
    geo_mod.node_group = node_group
    
    nodes = node_group.nodes
    links = node_group.links
    
    # Create interface
    node_group.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    node_group.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    
    # Input/Output
    input_node = nodes.new('NodeGroupInput')
    input_node.location = (-200, 0)
    
    output_node = nodes.new('NodeGroupOutput')
    output_node.location = (600, 0)
    
    # Instance on points
    instance_node = nodes.new('GeometryNodeInstanceOnPoints')
    instance_node.location = (200, 0)
    
    # Object info for base particle
    object_info = nodes.new('GeometryNodeObjectInfo')
    object_info.location = (0, -100)
    object_info.inputs['Object'].default_value = base_particle
    
    # Connect
    links.new(input_node.outputs['Geometry'], instance_node.inputs['Points'])
    links.new(object_info.outputs['Geometry'], instance_node.inputs['Instance'])
    links.new(instance_node.outputs['Instances'], output_node.inputs['Geometry'])
    
    print("✅ Geometry node setup complete")
    return geo_mod


def quick_particle_upgrade():
    """
    Quick upgrade function that checks for common particle system setups
    and automatically upgrades them to geometry
    """
    upgraded = False
    
    # Check for test_renderer
    if hasattr(bpy, 'test_renderer'):
        print("Found test_renderer")
        
        if hasattr(bpy, 'test_particles') and hasattr(bpy, 'test_weather'):
            upgrade_to_geometry_particles(
                bpy.test_renderer,
                bpy.test_particles,
                bpy.test_weather
            )
            upgraded = True
    
    # Check for particle objects by name
    particle_objects = [
        obj for obj in bpy.data.objects 
        if 'particle' in obj.name.lower() and obj.type == 'MESH'
    ]
    
    if particle_objects and not upgraded:
        print(f"Found {len(particle_objects)} particle objects")
        # Upgrade logic here if needed
    
    if not upgraded:
        print("❌ No particle system found to upgrade")
        print("   Run your particle demo first, then run this upgrade")
    
    return upgraded


# Convenience functions for console use
def show_particles():
    """Make all particle objects visible"""
    count = 0
    for obj in bpy.data.objects:
        if 'particle' in obj.name.lower():
            obj.hide_viewport = False
            obj.hide_render = False
            count += 1
    print(f"Made {count} particle objects visible")


def hide_particles():
    """Hide all particle objects"""
    count = 0
    for obj in bpy.data.objects:
        if 'particle' in obj.name.lower():
            obj.hide_viewport = True
            obj.hide_render = True
            count += 1
    print(f"Hidden {count} particle objects")


def adjust_particle_size(scale_factor):
    """Adjust the size of all particle instances"""
    base_particle = bpy.data.objects.get("BaseParticle")
    if base_particle:
        base_particle.scale *= scale_factor
        print(f"Scaled base particle by {scale_factor}x")
        
        # Update all instances if they share the mesh
        for obj in bpy.data.objects:
            if obj.data == base_particle.data and obj != base_particle:
                obj.scale = base_particle.scale
        
        print("✅ All particle instances updated")
    else:
        print("❌ No base particle found")


if __name__ == "__main__":
    # Auto-upgrade if running in Blender
    if "bpy" in locals():
        quick_particle_upgrade()
    else:
        print("This script must be run within Blender!")