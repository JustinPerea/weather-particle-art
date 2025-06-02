#!/usr/bin/env python3
"""
Easy Blender Integration Runner
Quick script to run the weather particle system in Blender

Usage:
1. Open Blender
2. In Python Console: exec(open('run_blender_integration.py').read())
3. Or from terminal: blender --python run_blender_integration.py
"""

import sys
import os

# Add project root to Python path
project_root = "/Users/justinperea/Documents/Art/Pulsaur/3D/Weather_API/weather_art_v3"
sys.path.insert(0, project_root)

print("""
╔══════════════════════════════════════════════════════════════╗
║           Weather Particle Art - Blender Integration         ║
║                     Chat 4: GPU Rendering                    ║
╚══════════════════════════════════════════════════════════════╝
""")

# Check if we're in Blender
if "bpy" not in sys.modules:
    print("ERROR: This script must be run within Blender!")
    print("\nTo run:")
    print("1. Open Blender")
    print("2. Open Python Console (Scripting workspace)")
    print("3. Type: exec(open('run_blender_integration.py').read())")
    sys.exit(1)

import bpy

# Import our integration
from src.blender.blender_integration import WeatherParticleBlenderIntegration
from src.blender.optimization_utils import PerformanceMonitor, BlenderMemoryOptimizer

def setup_blender_scene():
    """Prepare Blender for optimal particle rendering"""
    
    print("\n🔧 Setting up Blender scene...")
    
    # Clean existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Optimize viewport
    BlenderMemoryOptimizer.optimize_viewport_settings()
    
    # Set up workspace
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            # Set to rendered view
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'RENDERED'
    
    print("✅ Scene prepared")

def run_interactive_demo():
    """Run the interactive weather particle demo"""
    
    print("\n🌊 Starting Weather Particle System...")
    print("━" * 60)
    
    # Create integration with 100K particles
    integration = WeatherParticleBlenderIntegration(
        particle_count=100_000,
        demo_mode=True  # Use demo weather patterns
    )
    
    # Set up performance monitor
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    print("\n📊 Performance Monitor Active")
    print("   Watch the viewport for real-time FPS display")
    
    print("\n🎬 Animation Controls:")
    print("   SPACE - Play/Pause animation")
    print("   ← → - Step through frames")
    print("   ESC - Stop animation")
    
    print("\n🌤️  Weather Patterns (cycling every 5 seconds):")
    print("   1. Storm - Heavy turbulence")
    print("   2. Calm - Gentle flow")
    print("   3. Heat Wave - Rising thermal currents")
    print("   4. Fog - Dense, slow movement")
    print("   5. Hurricane - Violent circular motion")
    
    print("\n" + "━" * 60)
    print("✨ System running! Press SPACE to start animation")
    
    # Create animation loop
    def animation_loop():
        """Main animation update loop"""
        frame = 1
        while frame < 10000:  # Long running
            integration.update_frame(frame)
            frame += 1
            
            # Let Blender update viewport
            yield 1
    
    # Register timer for animation
    bpy.app.timers.register(
        lambda: next(animation_loop(), None),
        first_interval=0.016  # 60 FPS
    )
    
    return integration

def create_quick_commands():
    """Create quick command functions for easy testing"""
    
    commands = """
# Quick Commands for Python Console:

# Change particle count:
integration.particle_system = ViscousParticleSystem(particle_count=50_000)
integration.renderer = BlenderParticleRenderer(particle_count=50_000)

# Change weather pattern:
integration.current_weather = integration.demo_weather.get_pattern('hurricane')
integration.force_field = integration.physics_engine.generate_3d_force_field(integration.current_weather)

# Benchmark performance:
metrics = integration.renderer.benchmark_performance()
print(f"Average: {metrics['avg_update_ms']:.2f}ms")

# Save screenshot:
bpy.ops.render.render(write_still=True)
"""
    
    print("\n📝 Quick Commands:")
    print(commands)

def main():
    """Main entry point"""
    
    # Set up scene
    setup_blender_scene()
    
    # Run demo
    integration = run_interactive_demo()
    
    # Show quick commands
    create_quick_commands()
    
    # Store integration in Blender for console access
    bpy.integration = integration
    print("\n💡 TIP: Access the integration in console with 'bpy.integration'")
    
    return integration

if __name__ == "__main__":
    # Run when executed
    integration = main()
    
    # If not in background, set up viewport
    if not bpy.app.background:
        # Switch to 3D viewport
        bpy.context.area.type = 'VIEW_3D'
        
        # Frame the particle system
        bpy.ops.view3d.view_all()
        
        print("\n🎉 Ready! Press SPACE to play animation")