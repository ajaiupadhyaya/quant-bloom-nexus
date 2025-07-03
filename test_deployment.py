#!/usr/bin/env python3
"""
Test script to verify Quant Bloom Nexus Trading Terminal deployment
"""

import requests
import time
import sys
import os
from urllib.parse import urljoin

def test_health_endpoint(base_url):
    """Test the health endpoint"""
    try:
        response = requests.get(urljoin(base_url, "/health"), timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_api_docs(base_url):
    """Test the API documentation endpoint"""
    try:
        response = requests.get(urljoin(base_url, "/docs"), timeout=10)
        if response.status_code == 200:
            print("✅ API documentation accessible")
            return True
        else:
            print(f"❌ API documentation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API documentation error: {e}")
        return False

def test_frontend(base_url):
    """Test the frontend"""
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            print("✅ Frontend accessible")
            return True
        else:
            print(f"❌ Frontend failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend error: {e}")
        return False

def main():
    print("🧪 Testing Quant Bloom Nexus Trading Terminal Deployment")
    print("=" * 60)
    
    # Test backend
    print("\n🔧 Testing Backend (FastAPI)")
    backend_url = "http://localhost:8000"
    backend_ok = test_health_endpoint(backend_url) and test_api_docs(backend_url)
    
    # Test frontend
    print("\n🌐 Testing Frontend (React)")
    frontend_url = "http://localhost:3000"
    frontend_ok = test_frontend(frontend_url)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Deployment Test Summary")
    print("=" * 60)
    
    if backend_ok and frontend_ok:
        print("🎉 All tests passed! Your trading terminal is ready.")
        print("\n🌐 Access your application:")
        print(f"   Frontend: {frontend_url}")
        print(f"   Backend API: {backend_url}")
        print(f"   API Documentation: {backend_url}/docs")
        print(f"   Health Check: {backend_url}/health")
        return 0
    else:
        print("❌ Some tests failed. Please check the logs above.")
        if not backend_ok:
            print("   - Backend service may not be running")
        if not frontend_ok:
            print("   - Frontend service may not be running")
        print("\n🔧 Troubleshooting:")
        print("   1. Check if Docker containers are running: docker-compose ps")
        print("   2. View logs: docker-compose logs -f")
        print("   3. Restart services: docker-compose restart")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 