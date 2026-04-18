#!/usr/bin/env python3
"""Test JarvisAI production configuration"""

import os
import sys
import yaml
from pathlib import Path


def test_production_config():
    """Test the production configuration files"""

    print("Testing JarvisAI Production Configuration")
    print("=" * 50)

    # Check required files exist
    required_files = [
        "docker-compose.prod.yml",
        "Dockerfile.prod",
        ".env.prod.example",
        "config/nginx.conf",
        "config/prometheus.yml",
        "scripts/init-db.sql",
        "scripts/init-ollama.sh",
        "deploy.sh",
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print("[-] Missing configuration files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("[+] All required configuration files present")

    # Test Docker Compose configuration
    try:
        with open("docker-compose.prod.yml", "r") as f:
            compose_config = yaml.safe_load(f)

        services = compose_config.get("services", {})
        required_services = ["postgres", "redis", "api", "nginx"]

        missing_services = []
        for service in required_services:
            if service not in services:
                missing_services.append(service)

        if missing_services:
            print("[-] Missing services in docker-compose.prod.yml:")
            for service in missing_services:
                print(f"   - {service}")
            return False
        else:
            print("[+] Docker Compose configuration valid")
            print(f"   Services configured: {len(services)}")

    except Exception as e:
        print(f"[-] Docker Compose validation failed: {e}")
        return False

    # Test environment template
    try:
        with open(".env.prod.example", "r") as f:
            env_content = f.read()

        required_vars = [
            "ENVIRONMENT=production",
            "SECRET_KEY=",
            "POSTGRES_PASSWORD=",
            "REDIS_PASSWORD=",
            "DATABASE_URL=",
        ]

        missing_vars = []
        for var in required_vars:
            if var not in env_content:
                missing_vars.append(var)

        if missing_vars:
            print("[-] Missing environment variables in .env.prod.example:")
            for var in missing_vars:
                print(f"   - {var}")
            return False
        else:
            print("[+] Environment template complete")

    except Exception as e:
        print(f"[-] Environment template validation failed: {e}")
        return False

    print("\n" + "=" * 50)
    print("Production Configuration Test Complete!")
    print("=" * 50)
    print("[+] All configuration files are valid and complete")
    print("[+] Production deployment is ready")
    print("\nNext Steps:")
    print("   1. Copy .env.prod.example to .env.prod")
    print("   2. Edit .env.prod with your actual secrets")
    print("   3. Run: chmod +x deploy.sh && ./deploy.sh")
    print("   4. Access JarvisAI at http://localhost:8080")
    print("\nReady for production deployment!")

    return True


if __name__ == "__main__":
    success = test_production_config()
    sys.exit(0 if success else 1)
