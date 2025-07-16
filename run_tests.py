#!/usr/bin/env python3
"""
Test Runner Script
==================

Comprehensive test runner for the Enterprise Trading Bot.
Provides different test execution modes and reporting options.
"""

import argparse
import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime

def run_command(cmd, description=None):
    """Run a command and return success status"""
    if description:
        print(f"\n{'='*60}")
        print(f"🚀 {description}")
        print(f"{'='*60}")
    
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode == 0:
        print(f"✅ Success: {description or 'Command completed'}")
        return True
    else:
        print(f"❌ Failed: {description or 'Command failed'}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Enterprise Trading Bot Test Runner")
    
    # Test types
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    # Test options
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Fast mode (stop on first failure)")
    parser.add_argument("--slow", action="store_true", help="Run slow tests")
    
    # Quality checks
    parser.add_argument("--lint", action="store_true", help="Run linting")
    parser.add_argument("--security", action="store_true", help="Run security checks")
    parser.add_argument("--format", action="store_true", help="Format code")
    
    # CI mode
    parser.add_argument("--ci", action="store_true", help="Run in CI mode (all checks)")
    
    # Specific test files
    parser.add_argument("--file", help="Run specific test file")
    parser.add_argument("--marker", help="Run tests with specific marker")
    
    args = parser.parse_args()
    
    # Set default to all tests if no specific test type selected
    if not any([args.unit, args.integration, args.file, args.marker]):
        args.all = True
    
    print(f"""
    🧪 Enterprise Trading Bot Test Runner
    =====================================
    
    Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Test Directory: {Path.cwd()}
    Python Version: {sys.version.split()[0]}
    """)
    
    # Check if we're in the right directory
    if not Path("pytest.ini").exists():
        print("❌ pytest.ini not found. Are you in the project root?")
        return False
    
    success = True
    
    # Install dependencies if needed
    if not Path("venv").exists():
        print("⚠️  Virtual environment not found. Installing dependencies...")
        if not run_command("pip install -r requirements.txt", "Installing main dependencies"):
            success = False
        if not run_command("pip install -r requirements-test.txt", "Installing test dependencies"):
            success = False
    
    # Format code
    if args.format:
        success &= run_command("python -m black . --line-length=100", "Formatting code with Black")
        success &= run_command("python -m isort . --profile=black", "Sorting imports with isort")
    
    # Linting
    if args.lint or args.ci:
        success &= run_command(
            "python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics",
            "Running critical linting checks"
        )
        success &= run_command(
            "python -m flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics",
            "Running full linting analysis"
        )
    
    # Security checks
    if args.security or args.ci:
        success &= run_command("python -m bandit -r . -ll", "Running security analysis with Bandit")
        success &= run_command("python -m safety check", "Checking dependencies for vulnerabilities")
    
    # Build base pytest command
    base_cmd = "python -m pytest"
    
    # Add verbosity
    if args.verbose:
        base_cmd += " -v"
    else:
        base_cmd += " --tb=short"
    
    # Add parallel execution
    if args.parallel:
        base_cmd += " -n auto --dist worksteal"
    
    # Add fast mode
    if args.fast:
        base_cmd += " -x"
    
    # Add coverage
    if args.coverage or args.ci:
        base_cmd += " --cov=. --cov-report=html --cov-report=term-missing --cov-report=json"
    
    # Run specific test types
    if args.unit:
        cmd = f"{base_cmd} tests/unit/"
        success &= run_command(cmd, "Running unit tests")
    
    if args.integration:
        cmd = f"{base_cmd} tests/integration/"
        success &= run_command(cmd, "Running integration tests")
    
    if args.all:
        cmd = f"{base_cmd} tests/"
        success &= run_command(cmd, "Running all tests")
    
    if args.file:
        cmd = f"{base_cmd} {args.file}"
        success &= run_command(cmd, f"Running specific test file: {args.file}")
    
    if args.marker:
        cmd = f"{base_cmd} -m {args.marker}"
        success &= run_command(cmd, f"Running tests with marker: {args.marker}")
    
    if args.slow:
        cmd = f"{base_cmd} -m slow"
        success &= run_command(cmd, "Running slow tests")
    
    # Performance tests
    if args.ci:
        benchmark_cmd = f"{base_cmd} -k benchmark --benchmark-only --benchmark-json=benchmark.json"
        run_command(benchmark_cmd, "Running performance benchmarks")
    
    # Final report
    print(f"\n{'='*60}")
    print(f"📊 Test Execution Summary")
    print(f"{'='*60}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("✅ All tests passed successfully!")
        if args.coverage or args.ci:
            print("📈 Coverage report generated in htmlcov/")
    else:
        print("❌ Some tests failed!")
        return False
    
    # Open coverage report if requested
    if args.coverage and success:
        coverage_path = Path("htmlcov/index.html")
        if coverage_path.exists():
            print(f"📊 Coverage report: {coverage_path.absolute()}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)