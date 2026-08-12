"""
Quick test to verify all dependencies are working
"""

print("Testing dependencies...")

# Test 1: NumPy
import numpy as np
print("✓ NumPy imported")

# Test 2: Matplotlib
import matplotlib.pyplot as plt
print("✓ Matplotlib imported")

# Test 3: SlimeVolley
import slimevolleygym
print("✓ SlimeVolley imported")

# Test 4: Create environment
env = slimevolleygym.SlimeVolleyEnv()
print("✓ SlimeVolley environment created")

# Test 5: Basic environment interaction
obs = env.reset()
print(f"✓ Environment reset, observation shape: {obs.shape}")

# Test 6: Take a step
action = [0, 0, 0]  # [left/right, jump, no-op]
obs, reward, done, info = env.step(action)
print(f"✓ Environment step executed")

# Test 7: ImageIO
import imageio
print("✓ ImageIO imported")

print("\n🎉 ALL DEPENDENCIES WORKING!")
print("\nObservation space:", obs.shape)
print("Action space: 3 discrete actions (left/right, jump, no-op)")
print("\nReady to build NEAT! 🚀")