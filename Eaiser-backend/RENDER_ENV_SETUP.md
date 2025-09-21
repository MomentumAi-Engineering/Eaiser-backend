# 🚀 Render Environment Variables Setup Guide

## Required Environment Variables for Eaiser Backend Deployment

### 📋 Essential Configuration

#### 1. MongoDB Configuration
```bash
# Primary MongoDB URI (Atlas recommended for production)
MONGODB_URL=mongodb+srv://snapfix:Chrishabh100@snapfixcluster.7po8xxx.mongodb.net/snapfix?retryWrites=true&w=majority

# Alternative MongoDB URI (fallback)
MONGODB_URI=mongodb+srv://snapfix:Chrishabh100@snapfixcluster.7po8xxx.mongodb.net/snapfix?retryWrites=true&w=majority

# Database name (optional - extracted from URI if not provided)
MONGODB_NAME=snapfix
```

#### 2. Redis Configuration (Optional - App works without Redis)
```bash
# Redis host (leave empty for no Redis)
REDIS_HOST=

# Redis port (leave empty for no Redis)
REDIS_PORT=

# Redis password (if required)
REDIS_PASSWORD=

# Redis database number
REDIS_DB=0
```

#### 3. Application Configuration
```bash
# Environment type
ENVIRONMENT=production

# API Keys and Secrets
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# JWT Secret for authentication
JWT_SECRET=your_super_secure_jwt_secret_here

# CORS Origins (comma-separated)
CORS_ORIGINS=https://your-frontend-domain.com,https://www.your-frontend-domain.com

# Log level
LOG_LEVEL=INFO
```

### 🔧 Render Dashboard Setup Instructions

1. **Go to Render Dashboard** → Your Service → Environment

2. **Add Environment Variables:**
   ```
   MONGODB_URL = mongodb+srv://snapfix:Chrishabh100@snapfixcluster.7po8xxx.mongodb.net/snapfix?retryWrites=true&w=majority
   ENVIRONMENT = production
   LOG_LEVEL = INFO
   ```

3. **Optional Redis Variables** (leave empty if no Redis service):
   ```
   REDIS_HOST = 
   REDIS_PORT = 
   REDIS_PASSWORD = 
   REDIS_DB = 0
   ```

4. **Update Start Command** in Render Dashboard:
   ```bash
   cd Eaiser-backend/app && uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
   ```

### 🎯 Key Features of Updated Configuration

#### ✅ MongoDB Atlas Support
- **Production-ready connection** with MongoDB Atlas
- **Automatic SSL/TLS** detection for `mongodb+srv://` URIs
- **Connection pooling** optimized for Render resource limits
- **Retry logic** with exponential backoff
- **Graceful degradation** if MongoDB is unavailable

#### ✅ Redis Graceful Handling
- **Optional Redis** - app works without Redis
- **Automatic fallback** to database-only mode
- **No startup failures** if Redis is unavailable
- **Performance caching** when Redis is available

#### ✅ Production Optimizations
- **Connection timeouts** increased for production stability
- **Resource limits** optimized for Render environment
- **Comprehensive logging** for debugging
- **Error handling** with graceful degradation

### 🔍 Troubleshooting

#### Common Issues:

1. **MongoDB Connection Refused**
   - ✅ **Fixed**: Updated to use MongoDB Atlas URI
   - ✅ **Fixed**: Added retry logic and better error handling

2. **Redis Connection Refused**
   - ✅ **Fixed**: App continues without Redis
   - ✅ **Fixed**: Graceful fallback to database-only mode

3. **Environment Variables Not Set**
   - Check Render Dashboard → Environment tab
   - Ensure `MONGODB_URL` is set correctly
   - Redeploy after adding environment variables

### 📊 Expected Deployment Logs

#### ✅ Successful Startup:
```
INFO: Started server process [55]
INFO: Waiting for application startup.
🚀 Starting Eaiser AI backend server...
🔧 MongoDB Configuration:
   URI: mongodb+srv://snapfix...
   Database: snapfix
✅ Successfully connected to MongoDB database: snapfix
🔧 Connection pool: maxPoolSize=50, minPoolSize=5
🌐 MongoDB URI type: Atlas Cloud
⚠️ Redis unavailable - continuing without caching
✅ Eaiser AI backend started successfully (Redis unavailable)
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:10000
```

### 🚀 Next Steps After Environment Setup

1. **Add Environment Variables** in Render Dashboard
2. **Trigger Manual Deploy** or push new commit
3. **Monitor Deployment Logs** for successful startup
4. **Test API Endpoints** to verify functionality
5. **Check MongoDB Atlas** connection in logs

### 📝 Notes

- **MongoDB Atlas** is recommended for production
- **Redis is optional** - app works without it
- **Environment variables** must be set in Render Dashboard
- **Redeploy required** after adding environment variables
- **Connection pooling** optimized for Render resource limits

---

**Last Updated**: January 2025  
**Status**: Production Ready ✅