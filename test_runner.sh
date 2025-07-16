#!/bin/bash
# Test Runner Script for Enterprise Trading Bot
# ============================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found. Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    print_error "pytest not found. Installing test dependencies..."
    pip install pytest pytest-asyncio pytest-cov pytest-mock
fi

# Main test runner
print_status "🧪 Starting Enterprise Trading Bot Test Suite"
print_status "============================================="

# Parse command line arguments
case "${1:-all}" in
    "unit")
        print_status "Running unit tests only..."
        python -m pytest tests/unit/ -v --tb=short
        ;;
    "integration")
        print_status "Running integration tests only..."
        python -m pytest tests/integration/ -v --tb=short
        ;;
    "coverage")
        print_status "Running tests with coverage..."
        python -m pytest tests/ -v --tb=short --cov=. --cov-report=html --cov-report=term-missing
        ;;
    "quick")
        print_status "Running quick test validation..."
        python -m pytest tests/unit/test_risk_manager.py::TestRiskManager::test_validate_new_position_success -v
        ;;
    "all"|*)
        print_status "Running all tests..."
        python -m pytest tests/ -v --tb=short
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    print_success "✅ All tests passed!"
else
    print_error "❌ Some tests failed!"
    exit 1
fi

print_status "Test execution completed successfully!"