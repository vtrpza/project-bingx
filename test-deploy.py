#!/usr/bin/env python3
"""
Test script to verify deployment readiness
========================================

This script tests if all dependencies and configurations
are ready for Render deployment.
"""

import sys
import os
import importlib
import traceback

def test_python_version():
    """Test Python version compatibility"""
    print(f"🐍 Python version: {sys.version}")
    major, minor = sys.version_info[:2]
    
    if major == 3 and minor >= 11:
        print("✅ Python version compatible with Render")
        return True
    else:
        print("❌ Python version may not be compatible with Render")
        return False

def test_critical_imports():
    """Test critical imports for the trading bot"""
    critical_modules = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'pandas',
        'numpy',
        'ccxt',
        'structlog',
        'aiohttp',
        'cryptography',
        'websockets',
        'pytz',
        'cachetools',
        'backoff'
    ]
    
    print("\n📦 Testing critical imports...")
    passed = 0
    failed = 0
    
    for module_name in critical_modules:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {module_name} {version}")
            passed += 1
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"⚠️  {module_name}: {e}")
            failed += 1
    
    print(f"\n📊 Import Results: {passed} passed, {failed} failed")
    return failed == 0

def test_app_startup():
    """Test if the main app can start"""
    print("\n🚀 Testing app startup...")
    
    try:
        # Set demo mode to avoid API requirements
        os.environ['TRADING_MODE'] = 'demo'
        os.environ['LOG_LEVEL'] = 'INFO'
        
        # Import main components
        from main import app
        from config.settings import settings
        
        print(f"✅ Main app imported successfully")
        print(f"✅ Settings loaded: mode={settings.trading_mode}")
        print(f"✅ FastAPI app created")
        
        return True
        
    except Exception as e:
        print(f"❌ App startup failed: {e}")
        traceback.print_exc()
        return False

def test_required_files():
    """Test if required files exist"""
    print("\n📁 Testing required files...")
    
    required_files = [
        'main.py',
        'requirements-render.txt',
        'runtime.txt',
        'render.yaml',
        '.python-version',
        'config/settings.py',
        'core/exchange_manager.py',
        'core/trading_engine.py'
    ]
    
    missing = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            missing.append(file_path)
    
    if missing:
        print(f"\n❌ Missing files: {missing}")
        return False
    else:
        print(f"\n✅ All required files present")
        return True

def test_environment_variables():
    """Test environment variable configuration"""
    print("\n🔧 Testing environment variables...")
    
    # Test default values
    os.environ.setdefault('TRADING_MODE', 'demo')
    os.environ.setdefault('LOG_LEVEL', 'INFO')
    
    required_vars = ['TRADING_MODE', 'LOG_LEVEL']
    optional_vars = ['BINGX_API_KEY', 'BINGX_SECRET_KEY']
    
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}={value}")
        else:
            print(f"❌ {var} not set")
    
    for var in optional_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}=***")
        else:
            print(f"ℹ️  {var} not set (optional for demo)")
    
    return True

def main():
    """Run all tests"""
    print("🧪 Testing Render deployment readiness...")
    print("=" * 50)
    
    tests = [
        test_python_version,
        test_required_files,
        test_critical_imports,
        test_environment_variables,
        test_app_startup
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All tests passed! Ready for Render deployment.")
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())