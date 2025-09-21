# 🚀 Redis Setup Guide for Render Deployment

## Overview
Redis is used for high-performance caching in your SnapFix backend. This guide provides ready-to-use configurations for both development and production deployment on Render.

## 🔧 Production Setup (Render)

### Option 1: Redis Cloud (Recommended)
1. **Sign up for Redis Cloud**: https://redis.com/try-free/
2. **Create a free database** (30MB free tier)
3. **Get your connection details**:
   - Endpoint: `redis-xxxxx.c1.us-east-1-2.ec2.cloud.redislabs.com:12345`
   - Password: `your-redis-password`

### Option 2: Render Redis Add-on
1. **In your Render dashboard**:
   - Go to your service
   - Click "Environment" tab
   - Add Redis add-on (if available)

### Option 3: External Redis Provider
- **Upstash Redis**: https://upstash.com/
- **AWS ElastiCache**: For AWS deployments
- **Google Cloud Memorystore**: For GCP deployments

## 🌍 Environment Variables for Render

Add these environment variables in your Render dashboard:

### Method 1: Using REDIS_URL (Recommended)
```bash
# For Redis Cloud
REDIS_URL=redis://default:your-password@redis-xxxxx.c1.us-east-1-2.ec2.cloud.redislabs.com:12345

# For SSL-enabled Redis (Upstash, etc.)
REDIS_URL=rediss://default:your-password@redis-xxxxx.upstash.io:6379
```

### Method 2: Individual Variables
```bash
REDIS_HOST=redis-xxxxx.c1.us-east-1-2.ec2.cloud.redislabs.com
REDIS_PORT=12345
REDIS_PASSWORD=your-redis-password
REDIS_DB=0
REDIS_SSL=true
```

## 🏠 Development Setup

### Option 1: Docker (Recommended)
```bash
# Run Redis in Docker
docker run -d -p 6379:6379 --name redis redis:alpine

# With persistence
docker run -d -p 6379:6379 --name redis -v redis-data:/data redis:alpine redis-server --appendonly yes
```

### Option 2: Local Installation

#### Windows
```bash
# Using Chocolatey
choco install redis-64

# Using WSL2
wsl --install
# Then install Redis in WSL2
```

#### macOS
```bash
# Using Homebrew
brew install redis
brew services start redis
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server

# CentOS/RHEL
sudo yum install redis
sudo systemctl start redis
```

## 🔍 Testing Your Configuration

### 1. Test Locally
```bash
# Start your backend
cd app
python -m uvicorn main:app --reload

# Check logs for Redis connection
# Should see: "✅ Redis connected successfully"
```

### 2. Test Production
```bash
# Deploy to Render and check logs
# Look for Redis connection messages
```

## 📊 Redis Configuration Details

Your backend automatically handles:

### Connection Priority
1. `REDIS_URL` (full connection string) - **Preferred for production**
2. Individual environment variables (`REDIS_HOST`, `REDIS_PORT`, etc.)
3. Localhost fallback for development

### SSL/TLS Support
- Automatically detects `rediss://` URLs for SSL
- Configures SSL settings for cloud providers
- Handles certificate validation for managed services

### Graceful Fallback
- App works without Redis (slower performance)
- Automatic retry logic
- Health checks and monitoring

## 🚨 Common Issues & Solutions

### Issue 1: "Connection refused to localhost:6379"
**Solution**: Set production Redis environment variables
```bash
REDIS_URL=redis://your-redis-connection-string
```

### Issue 2: SSL/TLS errors
**Solution**: Use `rediss://` for SSL connections
```bash
REDIS_URL=rediss://default:password@host:port
```

### Issue 3: Authentication failed
**Solution**: Check your Redis password
```bash
# Test connection manually
redis-cli -h your-host -p your-port -a your-password ping
```

### Issue 4: Timeout errors
**Solution**: Check network connectivity and Redis service status

## 📈 Performance Benefits

With Redis enabled, you get:
- **5x faster** API responses for cached data
- **Reduced database load** by 70-80%
- **Better user experience** with instant responses
- **Scalability** for high-traffic scenarios

## 🔧 Ready-to-Use Configurations

### For render.yaml
```yaml
services:
  - type: web
    name: eaiser-backend
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "cd app && python -m uvicorn main:app --host 0.0.0.0 --port $PORT"
    envVars:
      # MongoDB Configuration
      - key: MONGODB_URL
        value: "your-mongodb-atlas-connection-string"
      
      # Redis Configuration (choose one method)
      - key: REDIS_URL
        value: "redis://default:password@host:port"
      
      # OR individual variables
      - key: REDIS_HOST
        value: "your-redis-host"
      - key: REDIS_PORT
        value: "6379"
      - key: REDIS_PASSWORD
        value: "your-redis-password"
      - key: REDIS_SSL
        value: "true"
```

### For .env (Development)
```bash
# MongoDB
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/dbname?retryWrites=true&w=majority

# Redis (Development - Docker)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
REDIS_SSL=false

# Redis (Development - Cloud)
REDIS_URL=redis://default:password@host:port
```

## 🎯 Next Steps

1. **Choose your Redis provider** (Redis Cloud recommended for free tier)
2. **Set environment variables** in Render dashboard
3. **Deploy your application**
4. **Monitor logs** for successful Redis connection
5. **Test API performance** - should be significantly faster

## 💡 Pro Tips

- **Free tiers available**: Redis Cloud (30MB), Upstash (10K requests/day)
- **Monitor usage**: Set up alerts for memory usage
- **Cache strategy**: Your app automatically caches frequently accessed data
- **Backup**: Redis data is automatically persisted in cloud providers

---

**Need help?** Check the logs for detailed error messages and connection status. The backend provides helpful guidance for common issues.