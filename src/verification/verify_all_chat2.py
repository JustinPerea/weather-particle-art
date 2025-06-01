#!/usr/bin/env python3
"""
Chat 2 Complete Verification
Runs all weather and physics tests for Issues #1 and #2
"""

import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def run_verification_script(script_path: str, description: str) -> bool:
    """Run a verification script and return success status"""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent  # Run from project root
        )
        
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Failed to run {description}: {e}")
        return False

def main():
    """Run all Chat 2 verification tests"""
    
    print("=" * 60)
    print("CHAT 2 COMPLETE VERIFICATION")
    print("Weather Data & Physics Force Fields")
    print("=" * 60)
    
    # Define verification scripts
    verifications = [
        ("src/verification/verify_weather_api.py", "Weather API Verification (Issue #1)"),
        ("src/verification/verify_physics_engine.py", "Physics Engine Verification (Issue #2)")
    ]
    
    results = []
    
    # Run each verification
    for script_path, description in verifications:
        success = run_verification_script(script_path, description)
        results.append((description, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{description}: {status}")
        if not success:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ ALL CHAT 2 VERIFICATIONS PASSED!")
        print("\n📋 Deliverables created:")
        print("  - src/weather/noaa_api.py - Weather API with caching")
        print("  - src/physics/force_field_engine.py - Physics force field generator")
        print("  - verification_outputs/chat_2_physics/ - All verification plots and data")
        print("\n🚀 Ready for Chat 3 - Particle System Integration!")
        print("\n📎 Handoff files for Chat 3:")
        print("  - verification_outputs/chat_2_physics/chat2_to_chat3_handoff.json")
        print("  - verification_outputs/chat_2_physics/sample_weather_data.json")
        print("  - verification_outputs/chat_2_physics/sample_force_fields.json")
        return 0
    else:
        print("\n⚠️  Some verifications failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())