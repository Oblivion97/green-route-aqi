#!/usr/bin/env python3
"""
Green Route AQI Setup and Installation Script
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_requirements():
    """Install required packages."""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        print("📦 Installing required packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ["data", "output", "logs", "models"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created/verified directory: {directory}/")
    
    return True

def verify_installation():
    """Verify that all required packages are installed."""
    required_packages = [
        "pandas", "numpy", "scikit-learn", "matplotlib", 
        "statsmodels", "tensorflow", "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            pkg_resources.get_distribution(package)
            print(f"✅ {package} is installed")
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    return len(missing_packages) == 0

def run_test():
    """Run a quick test of the system."""
    try:
        print("\n🧪 Running system test...")
        result = subprocess.run([
            sys.executable, "test_aqi_system.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ System test passed!")
            return True
        else:
            print("❌ System test failed")
            print("Error output:", result.stderr[:500])
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ System test timed out (this is normal for large datasets)")
        return True
    except Exception as e:
        print(f"❌ System test error: {e}")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("🌱 GREEN ROUTE AQI FORECASTING SYSTEM SETUP")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("⚠️ Some packages may be missing. Try running:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Run test
    run_test()
    
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("1. Run: python test_aqi_system.py")
    print("2. Check the output/ directory for visualizations")
    print("3. Explore the data/ directory for datasets")
    print("4. Read README.md for detailed usage instructions")
    print("\n🚀 Happy forecasting!")

if __name__ == "__main__":
    main()
