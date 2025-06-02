#!/usr/bin/env python3
"""
Enhanced Particle Rendering with Actual Geometry
Particles rendered as small spheres to show emission effects
"""

import bpy
import numpy as np
import time
from mathutils import Vector

class GeometryParticleRenderer:
    """Particle renderer using instanced geometry for visible emission"""
    
    def __init__(self, particle_count=100_000):
        self.particle_count = particle_count
        self.base_particle = None
        self.particle_system = None
        self.container = None
        self.material = None
        self.update_times = []
        
    def setup_particle_geometry(self):
        """Create base particle geometry and instancing system"""
        print(f"Setting up geometry-based particles for {self.particle_count:,} instances...")
        
        # Create base particle (small icosphere)
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=1,
            radius=0.01,  # Small sphere
            location=(0, 0, -10)  # Hide it below scene
        )
        self.base_particle = bpy.context.active_object
        self.base_particle.name = "BaseParticle"
        
        # Create container for instances
        container_mesh = bpy.data.meshes.new("ParticleContainer")
        self.container = bpy.data.objects.new("ParticleContainer", container_mesh)
        bpy.context.collection.objects.link(self.container)
        
        # Setup geometry nodes for instancing
        self._setup_geometry_nodes()
        
        print("✅ Geometry particle system created")
    
    def _setup_geometry_nodes(self):
        """Setup geometry nodes for efficient particle instancing"""
        # Add geometry nodes modifier
        geo_modifier = self.container.modifiers.new("ParticleInstancer", 'NODES')
        
        # Create node group
        node_group = bpy.data.node_groups.new("ParticleInstancer", 'GeometryNodeTree')
        geo_modifier.node_group = node_group
        
        nodes = node_group.nodes
        links = node_group.links
        
        # Create interface
        node_group.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket('Points', in_out='INPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket('Instance', in_out='INPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket('Scale', in_out='INPUT', socket_type='NodeSocketFloat')
        node_group.interface.items_tree[-1].default_value = 1.0
        
        # Input/Output nodes
        input_node = nodes.new('NodeGroupInput')
        input_node.location = (-400, 0)
        
        output_node = nodes.new('NodeGroupOutput')
        output_node.location = (400, 0)
        
        # Instance on points node
        instance_node = nodes.new('GeometryNodeInstanceOnPoints')
        instance_node.location = (0, 0)
        
        # Object info for base particle
        object_info = nodes.new('GeometryNodeObjectInfo')
        object_info.location = (-200, -100)
        object_info.inputs['Object'].default_value = self.base_particle
        
        # Scale instances
        scale_node = nodes.new('GeometryNodeScaleInstances')
        scale_node.location = (200, 0)
        
        # Connect nodes
        links.new(input_node.outputs['Points'], instance_node.inputs['Points'])
        links.new(object_info.outputs['Geometry'], instance_node.inputs['Instance'])
        links.new(instance_node.outputs['Instances'], scale_node.inputs['Instances'])
        links.new(input_node.outputs['Scale'], scale_node.inputs['Scale'])
        links.new(scale_node.outputs['Instances'], output_node.inputs['Geometry'])
    
    def apply_material(self, material):
        """Apply material to base particle"""
        if self.base_particle and material:
            self.base_particle.data.materials.clear()
            self.base_particle.data.materials.append(material)
            self.material = material
            print(f"✅ Applied material: {material.name}")
    
    def update_from_vertices(self, vertex_positions):
        """Update particle instances from vertex positions"""
        start_time = time.perf_counter()
        
        # Create points for instancing
        if not self.container.data.vertices:
            # First time - create vertices
            self.container.data.vertices.add(len(vertex_positions))
        
        # Update vertex positions
        self.container.data.vertices.foreach_set("co", vertex_positions.flatten())
        self.container.data.update()
        
        update_time = (time.perf_counter() - start_time) * 1000
        self.update_times.append(update_time)
        
        if len(self.update_times) % 60 == 0 and self.update_times:
            avg_ms = np.mean(self.update_times[-60:])
            print(f"  Geometry update: {avg_ms:.2f}ms")
    
    def set_particle_scale(self, scale):
        """Adjust particle size"""
        if self.container and self.container.modifiers:
            geo_mod = self.container.modifiers.get("ParticleInstancer")
            if geo_mod and geo_mod.node_group:
                # Find scale input
                for input in geo_mod.node_group.interface.items_tree:
                    if input.name == "Scale":
                        input.default_value = scale
                        break


def update_renderer_with_geometry(renderer, material_system):
    """Update existing renderer to use geometry particles"""
    
    # Create geometry particle system
    geo_renderer = GeometryParticleRenderer(renderer.particle_count)
    geo_renderer.setup_particle_geometry()
    
    # Apply the Anadol material
    if material_system and material_system.material:
        geo_renderer.apply_material(material_system.material)
    
    # Hide original vertex-based particles
    if hasattr(renderer, 'obj'):
        renderer.obj.hide_viewport = True
        renderer.obj.hide_render = True
    
    # Add geometry renderer to the main renderer
    renderer.geo_renderer = geo_renderer
    
    # Override update method
    original_update = renderer.update_particles
    
    def update_with_geometry(positions, weather_data=None):
        # Call original update
        original_update(positions, weather_data)
        
        # Update geometry instances
        if hasattr(renderer, 'geo_renderer'):
            renderer.geo_renderer.update_from_vertices(positions)
    
    renderer.update_particles = update_with_geometry
    
    return geo_renderer


# Convenience function to upgrade existing demo
def upgrade_to_geometry_particles():
    """Upgrade the current demo to use geometry particles"""
    if hasattr(bpy, 'test_renderer') and hasattr(bpy.test_renderer, 'material_system'):
        print("\n=== UPGRADING TO GEOMETRY PARTICLES ===")
        
        # Upgrade the renderer
        geo_renderer = update_renderer_with_geometry(
            bpy.test_renderer, 
            bpy.test_renderer.material_system
        )
        
        # Adjust particle size based on count
        particle_count = bpy.test_renderer.particle_count
        if particle_count > 50000:
            scale = 0.5  # Smaller for many particles
        elif particle_count > 10000:
            scale = 0.7
        else:
            scale = 1.0
            
        geo_renderer.set_particle_scale(scale)
        
        print(f"✅ Upgraded to geometry particles with scale {scale}")
        print("   Particles now render with visible emission!")
        
        # Update once to show current state
        if hasattr(bpy, 'test_particles'):
            bpy.test_renderer.update_particles(
                bpy.test_particles.positions,
                bpy.test_weather
            )
        
        return geo_renderer
    else:
        print("❌ No active demo found. Run the merged_standalone_demo.py first!")
        return None


if __name__ == "__main__":
    # Auto-upgrade if running after demo
    upgrade_to_geometry_particles()