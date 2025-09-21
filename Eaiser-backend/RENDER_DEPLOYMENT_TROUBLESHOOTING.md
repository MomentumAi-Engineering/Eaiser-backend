# 🚨 Render Deployment Troubleshooting Guide

## Current Issues Analysis

### 1. CORS Error
```
Access to fetch at 'https://eaiser-backend.onrender.com/api/authorities/37062' from origin 'https://www.eaiser.ai' has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

**Status**: ✅ **RESOLVED** - CORS is properly configured in `main.py`

### 2. MongoDB Connection Error
```
Database initialization failed: Failed to connect to MongoDB after 3 attempts: localhost:27017: [Errno 111] Connection refused
```

**Status**: 🔧 **IN PROGRESS** - MongoDB Atlas connection needs environment variable setup

## 🔧 Immediate Solutions

### Step 1: Set MongoDB Environment Variable in Render

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Select your service**: `eaiser-backend`
3. **Go to Environment tab**
4. **Add/Update these variables**:

```bash
# Required - MongoDB Atlas Connection
MONGODB_URL=mongodb+srv://snapfix:Chrishabh100@snapfixcluster.7po8xxx.mongodb.net/eaiser?retryWrites=true&w=majority

# Optional - Alternative variable names (for compatibility)
MONGODB_URI=mongodb+srv://snapfix:Chrishabh100@snapfixcluster.7po8xxx.mongodb.net/eaiser?retryWrites=true&w=majority
MONGO_URI=mongodb+srv://snapfix:Chrishabh100@snapfixcluster.7po8xxx.mongodb.net/eaiser?retryWrites=true&w=majority

# Database name
MONGODB_NAME=eaiser

# API Keys (if needed)
GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 2: Verify MongoDB Atlas Configuration

1. **Check MongoDB Atlas**:
   - Cluster: `snapfixcluster`
   - Database: `eaiser`
   - User: `snapfix` with password `Chrishabh100`

2. **Network Access**:
   - Ensure `0.0.0.0/0` is whitelisted (allow all IPs)
   - Or add Render's IP ranges

3. **Database User Permissions**:
   - User should have `readWrite` access to `eaiser` database

### Step 3: Deploy and Monitor

1. **Trigger Deployment**:
   - Save environment variables in Render
   - This will automatically trigger a new deployment

2. **Monitor Logs**:
   - Watch deployment logs for MongoDB connection success
   - Look for: `✅ Successfully connected to MongoDB database: eaiser`

## 🔍 Expected Log Output (Success)

```
🔧 MongoDB Configuration:
   URI: mongodb+srv://***:***@snapfixcluster.7po8xxx.mongodb.net/eaiser
   Database: eaiser
   Environment: Production (Atlas)
🔄 Attempting to connect to MongoDB...
🔧 URI Type: Atlas Cloud
🔒 SSL/TLS enabled for secure connection
🔄 Connection attempt 1/5...
✅ MongoDB ping successful on attempt 1
✅ Successfully connected to MongoDB database: eaiser
📊 Found X collections in database
🔧 Connection pool: maxPoolSize=20, minPoolSize=2
📇 Database indexes created/verified successfully
```

## 🚨 Common Error Patterns & Solutions

### Error: "Authentication failed"
```bash
# Solution: Check username/password in connection string
MONGODB_URL=mongodb+srv://correct_username:correct_password@cluster.mongodb.net/eaiser
```

### Error: "Connection timeout"
```bash
# Solution: Check network access in MongoDB Atlas
# Whitelist 0.0.0.0/0 or specific Render IP ranges
```

### Error: "Database not found"
```bash
# Solution: Ensure database name matches
MONGODB_NAME=eaiser
# And connection string points to correct database
```

## 🔧 Environment Variables Priority

The application checks environment variables in this order:
1. `MONGODB_URL` (Render standard)
2. `MONGODB_URI` (Alternative)
3. `MONGO_URI` (Fallback)
4. `mongodb://localhost:27017` (Development fallback)

## 🧪 Testing After Deployment

### 1. Health Check
```bash
curl https://eaiser-backend.onrender.com/
# Expected: {"message": "SnapFix AI backend is up and running!"}
```

### 2. Database Connection Test
```bash
curl https://eaiser-backend.onrender.com/api/authorities/37062
# Should return authority data, not 500 error
```

### 3. CORS Test
- Open browser console on https://www.eaiser.ai
- Try API call - should work without CORS errors

## 📋 Deployment Checklist

- [ ] MongoDB Atlas cluster is running
- [ ] Database user has correct permissions
- [ ] Network access allows all IPs (0.0.0.0/0)
- [ ] `MONGODB_URL` environment variable is set in Render
- [ ] `MONGODB_NAME=eaiser` is set
- [ ] Deployment triggered and completed
- [ ] Logs show successful MongoDB connection
- [ ] API endpoints return data (not 500 errors)
- [ ] Frontend can access API without CORS errors

## 🆘 Emergency Fallback

If MongoDB Atlas is not working, the application will:
1. Start without database connection
2. Log detailed error messages
3. Continue running (graceful degradation)
4. Return 500 errors for database operations

This allows you to:
1. Debug connection issues
2. Fix environment variables
3. Restart without full redeployment

## 📞 Next Steps

1. **Set environment variables in Render**
2. **Wait for automatic deployment**
3. **Check logs for MongoDB connection success**
4. **Test API endpoints**
5. **Verify frontend integration**

---

**Last Updated**: January 2025
**Status**: MongoDB connection fix in progress