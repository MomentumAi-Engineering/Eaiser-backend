#!/usr/bin/env python3
"""
Environment Variables Validation Script
Checks if all required environment variables are properly configured for deployment.
"""

import os
from dotenv import load_dotenv

def check_environment_variables():
    """Check if all required environment variables are present and valid."""
    
    # Load environment variables from .env file
    load_dotenv('app/.env')
    
    print("🔍 Checking Environment Variables for Deployment Readiness...")
    print("=" * 60)
    
    # Critical environment variables for production
    required_vars = {
        'MONGO_URI': 'MongoDB connection string',
        'MONGODB_NAME': 'MongoDB database name',
        'GOOGLE_API_KEY': 'Google API key for AI services',
        'SENDGRID_API_KEY': 'SendGrid API key for email services',
        'EMAIL_USER': 'Email user for notifications',
        'EMAIL_HOST': 'SMTP host for email',
        'EMAIL_PORT': 'SMTP port for email'
    }
    
    # Optional but recommended variables
    optional_vars = {
        'REDIS_HOST': 'Redis host (defaults to localhost)',
        'REDIS_PORT': 'Redis port (defaults to 6379)',
        'REDIS_PASSWORD': 'Redis password (if required)',
        'JWT_SECRET_KEY': 'JWT secret key for authentication',
        'ENVIRONMENT': 'Environment type (production/development)'
    }
    
    missing_required = []
    missing_optional = []
    
    print("📋 Required Environment Variables:")
    print("-" * 40)
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if 'KEY' in var or 'PASSWORD' in var or 'URI' in var:
                display_value = f"{value[:10]}...{value[-4:]}" if len(value) > 14 else "***"
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
        else:
            print(f"❌ {var}: NOT SET - {description}")
            missing_required.append(var)
    
    print("\n📋 Optional Environment Variables:")
    print("-" * 40)
    
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            if 'KEY' in var or 'PASSWORD' in var:
                display_value = f"{value[:6]}..." if len(value) > 6 else "***"
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
        else:
            print(f"⚠️  {var}: NOT SET - {description}")
            missing_optional.append(var)
    
    print("\n" + "=" * 60)
    print("📊 DEPLOYMENT READINESS SUMMARY:")
    print("=" * 60)
    
    if not missing_required:
        print("✅ All required environment variables are configured!")
        print("🚀 Backend is ready for deployment to production!")
        
        if missing_optional:
            print(f"\n⚠️  {len(missing_optional)} optional variables are missing:")
            for var in missing_optional:
                print(f"   - {var}")
            print("   (These are optional but recommended for production)")
        
        return True
    else:
        print(f"❌ {len(missing_required)} required environment variables are missing:")
        for var in missing_required:
            print(f"   - {var}")
        print("\n🛑 Backend is NOT ready for deployment!")
        print("   Please set the missing required variables before deploying.")
        return False

if __name__ == "__main__":
    is_ready = check_environment_variables()
    exit(0 if is_ready else 1)