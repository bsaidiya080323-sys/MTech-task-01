"""
Charity Fundraising Campaign Optimizer
======================================
An AI-powered system for optimizing charity fundraising campaigns using
machine learning and operations research techniques.

Author: Noor ul Huda
Reg No: Mtech-AI26092
Project: Charity Fundraising Campaign Optimizer
"""

import sys
import os

def check_dependencies():
    """Check if all required packages are installed."""
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'PIL': 'Pillow',
        'scipy': 'scipy'
    }

    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("❌ Missing dependencies. Please install them using:")
        print(f"   pip install {' '.join(missing)}")
        print("
Or install all at once:")
        print("   pip install -r requirements.txt")
        return False

    return True

def main():
    """Main entry point."""
    print("=" * 70)
    print("🎯 CHARITY FUNDRAISING CAMPAIGN OPTIMIZER")
    print("=" * 70)
    print("AI-Powered Campaign Optimization for Maximum Impact")
    print("Author: Noor ul Huda | Reg No: Mtech-AI26092")
    print("=" * 70)
    print()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    print("✅ All dependencies satisfied!")
    print("🚀 Launching application...")
    print()

    # Launch GUI
    try:
        from gui import main as gui_main
        gui_main()
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
