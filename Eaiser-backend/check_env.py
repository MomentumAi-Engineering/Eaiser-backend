#!/usr/bin/env python3
"""
Environment Variables Validation Script for Eaiser Backend
Validates all required and optional environment variables for production deployment.
"""

import os
from dotenv import load_dotenv
from typing import Dict, List, Tuple

# Load environment variables from .env file if it exists
load_dotenv()

def mask_sensitive_value(value: str, show_chars: int = 4) -> str:
    """
    Mask sensitive values showing only first few characters
    
    Args:
        value: The sensitive value to mask
        show_chars: Number of characters to show at the beginning
    
    Returns:
        Masked string
    """
    if len(value) <= show_chars:
        return "*" * len(value)
    return value[:show_chars] + "*" * (len(value) - show_chars)

def check_environment_variables() -> Tuple[Dict[str, bool], Dict[str, bool]]:
    """
    Check all required and optional environment variables
    
    Returns:
        Tuple of (required_vars_status, optional_vars_status)
    """
    
    # Required environment variables for production
    required_vars = {
        'MONGO_URI': 'MongoDB connection string',
        'MONGODB_NAME': 'MongoDB database name',
        'GOOGLE_API_KEY': 'Google API key for geocoding',
        'SENDGRID_API_KEY': 'SendGrid API key for email',
        'EMAIL_USER': 'Email user for sending notifications',
        'EMAIL_HOST': 'SMTP host for email',
        'EMAIL_PORT': 'SMTP port for email'
    }
    
    # Optional environment variables (nice to have)
    optional_vars = {
        'REDIS_HOST': 'Redis host for caching',
        'REDIS_PORT': 'Redis port',
        'REDIS_PASSWORD': 'Redis password',
        'JWT_SECRET_KEY': 'JWT secret key for authentication',
        'ENVIRONMENT': 'Environment type (development/production)',
        'OPENAI_API_KEY': 'OpenAI API key for AI features',
        'GEMINI_API_KEY': 'Gemini API key for AI features'
    }
    
    print("🔍 Environment Variables Validation")
    print("=" * 50)
    
    # Check required variables
    print("\n📋 Required Variables:")
    required_status = {}
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            required_status[var] = True
            masked_value = mask_sensitive_value(value) if 'KEY' in var or 'PASSWORD' in var or 'URI' in var else value
            print(f"✅ {var}: {masked_value} ({description})")
        else:
            required_status[var] = False
            print(f"❌ {var}: NOT SET ({description})")
    
    # Check optional variables
    print("\n📋 Optional Variables:")
    optional_status = {}
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            optional_status[var] = True
            masked_value = mask_sensitive_value(value) if 'KEY' in var or 'PASSWORD' in var else value
            print(f"✅ {var}: {masked_value} ({description})")
        else:
            optional_status[var] = False
            print(f"⚠️  {var}: NOT SET ({description})")
    
    return required_status, optional_status

def validate_specific_formats() -> Dict[str, bool]:
    """
    Validate specific format requirements for certain variables
    
    Returns:
        Dictionary of validation results
    """
    validations = {}
    
    # Validate MongoDB URI format
    mongo_uri = os.getenv('MONGO_URI')
    if mongo_uri:
        if mongo_uri.startswith('mongodb://') or mongo_uri.startswith('mongodb+srv://'):
            validations['mongo_uri_format'] = True
            print("✅ MongoDB URI format is valid")
        else:
            validations['mongo_uri_format'] = False
            print("❌ MongoDB URI format is invalid (should start with mongodb:// or mongodb+srv://)")
    else:
        validations['mongo_uri_format'] = False
    
    # Validate email format
    email_user = os.getenv('EMAIL_USER')
    if email_user:
        if '@' in email_user and '.' in email_user:
            validations['email_format'] = True
            print("✅ Email format is valid")
        else:
            validations['email_format'] = False
            print("❌ Email format is invalid")
    else:
        validations['email_format'] = False
    
    # Validate Redis port if provided
    redis_port = os.getenv('REDIS_PORT')
    if redis_port:
        try:
            port_num = int(redis_port)
            if 1 <= port_num <= 65535:
                validations['redis_port_format'] = True
                print("✅ Redis port is valid")
            else:
                validations['redis_port_format'] = False
                print("❌ Redis port is out of valid range (1-65535)")
        except ValueError:
            validations['redis_port_format'] = False
            print("❌ Redis port is not a valid number")
    else:
        validations['redis_port_format'] = True  # Not required
    
    return validations

def generate_deployment_summary(required_status: Dict[str, bool], 
                              optional_status: Dict[str, bool], 
                              validations: Dict[str, bool]) -> None:
    """
    Generate deployment readiness summary
    
    Args:
        required_status: Status of required variables
        optional_status: Status of optional variables
        validations: Format validation results
    """
    print("\n" + "=" * 50)
    print("📊 DEPLOYMENT READINESS SUMMARY")
    print("=" * 50)
    
    # Count required variables
    required_set = sum(required_status.values())
    total_required = len(required_status)
    
    # Count optional variables
    optional_set = sum(optional_status.values())
    total_optional = len(optional_status)
    
    # Count validations passed
    validations_passed = sum(validations.values())
    total_validations = len(validations)
    
    print(f"Required Variables: {required_set}/{total_required}")
    print(f"Optional Variables: {optional_set}/{total_optional}")
    print(f"Format Validations: {validations_passed}/{total_validations}")
    
    # Deployment readiness
    if required_set == total_required and validations_passed == total_validations:
        print("\n🎉 DEPLOYMENT READY!")
        print("✅ All required environment variables are configured")
        print("✅ All format validations passed")
        print("🚀 Backend is ready for production deployment")
    else:
        print("\n⚠️  DEPLOYMENT NOT READY")
        if required_set < total_required:
            missing_required = [var for var, status in required_status.items() if not status]
            print(f"❌ Missing required variables: {', '.join(missing_required)}")
        
        if validations_passed < total_validations:
            failed_validations = [var for var, status in validations.items() if not status]
            print(f"❌ Failed validations: {', '.join(failed_validations)}")
        
        print("🔧 Please fix the above issues before deploying")
    
    # Optional improvements
    if optional_set < total_optional:
        missing_optional = [var for var, status in optional_status.items() if not status]
        print(f"\n💡 Optional improvements: {', '.join(missing_optional)}")

def main():
    """Main function to run environment validation"""
    print("🚀 Eaiser Backend - Environment Validation")
    print("Version: 1.0.0")
    print("Date: 2024")
    
    try:
        # Check environment variables
        required_status, optional_status = check_environment_variables()
        
        # Validate formats
        print("\n🔍 Format Validations:")
        validations = validate_specific_formats()
        
        # Generate summary
        generate_deployment_summary(required_status, optional_status, validations)
        
        # Exit with appropriate code
        all_required_set = all(required_status.values())
        all_validations_passed = all(validations.values())
        
        if all_required_set and all_validations_passed:
            exit(0)  # Success
        else:
            exit(1)  # Failure
            
    except Exception as e:
        print(f"\n❌ Error during validation: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()