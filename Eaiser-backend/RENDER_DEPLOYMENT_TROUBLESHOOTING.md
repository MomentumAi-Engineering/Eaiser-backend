# Render Deployment Troubleshooting Guide

## Common Issues and Solutions

### 1. Build Failures

#### Issue: Python Version Mismatch
```
Error: Python version not supported
```

**Solution:**
- Check `runtime.txt` contains: `python-3.11.9`
- Ensure requirements.txt is compatible with Python 3.11

#### Issue: Dependencies Installation Failed
```
Error: Could not install packages due to an EnvironmentError
```

**Solution:**
- Review `requirements.txt` for conflicting versions
- Remove version pins that cause conflicts
- Use `pip install --no-cache-dir` in Dockerfile if needed

### 2. Environment Variables Issues

#### Issue: Missing Environment Variables
```
Error: Environment variable 'MONGO_URI' not found
```

**Solution:**
1. Go to Render Dashboard → Service → Environment
2. Add all required environment variables:
   - `MONGO_URI`
   - `MONGODB_NAME`
   - `GOOGLE_API_KEY`
   - `SENDGRID_API_KEY`
   - `EMAIL_USER`

#### Issue: Environment Variables Not Loading
```
Error: Configuration not found
```

**Solution:**
- Ensure environment variables are set in Render dashboard
- Check variable names match exactly (case-sensitive)
- Restart the service after adding variables

### 3. Database Connection Issues

#### Issue: MongoDB Connection Timeout
```
Error: ServerSelectionTimeoutError: connection timeout
```

**Solution:**
1. Verify MongoDB Atlas connection string
2. Check IP whitelist includes `0.0.0.0/0` for Render
3. Ensure SSL/TLS is enabled in connection string
4. Test connection string locally first

#### Issue: Authentication Failed
```
Error: Authentication failed
```

**Solution:**
- Verify username/password in connection string
- Check database user permissions
- Ensure database name exists

### 4. Redis Connection Issues

#### Issue: Redis Connection Refused
```
Error: Connection refused to Redis server
```

**Solution:**
- Verify Redis host and port
- Check Redis password
- Ensure Redis instance is running
- Test Redis connection locally

#### Issue: Redis SSL/TLS Issues
```
Error: SSL connection failed
```

**Solution:**
- Set `REDIS_SSL=true` for production
- Verify Redis provider supports SSL
- Check certificate validity

### 5. Application Startup Issues

#### Issue: Port Binding Error
```
Error: Port already in use
```

**Solution:**
- Render automatically assigns port via `$PORT` environment variable
- Ensure application binds to `0.0.0.0:$PORT`
- Check `render.yaml` configuration

#### Issue: Import Errors
```
Error: ModuleNotFoundError
```

**Solution:**
- Verify all dependencies in `requirements.txt`
- Check Python path configuration
- Ensure proper package structure

### 6. Performance Issues

#### Issue: Slow Response Times
```
Warning: Response time > 30 seconds
```

**Solution:**
- Optimize database queries
- Implement caching with Redis
- Use connection pooling
- Consider upgrading Render plan

#### Issue: Memory Limit Exceeded
```
Error: Process killed due to memory limit
```

**Solution:**
- Optimize memory usage in application
- Upgrade to higher memory plan
- Implement proper garbage collection

### 7. SSL/HTTPS Issues

#### Issue: SSL Certificate Error
```
Error: SSL certificate verification failed
```

**Solution:**
- Render provides automatic SSL
- Ensure custom domain is properly configured
- Check DNS settings

### 8. Logging and Monitoring

#### Issue: No Logs Visible
```
Warning: Application logs not showing
```

**Solution:**
- Check logging configuration in application
- Use `print()` statements for debugging
- Verify log level settings

#### Issue: Application Crashes
```
Error: Application exited with code 1
```

**Solution:**
1. Check Render logs for error details
2. Test application locally
3. Verify all environment variables
4. Check database connections

### 9. Deployment Configuration

#### Issue: render.yaml Not Working
```
Error: Invalid render.yaml configuration
```

**Solution:**
- Validate YAML syntax
- Check indentation (use spaces, not tabs)
- Verify service type and settings
- Reference Render documentation

#### Issue: Build Command Failed
```
Error: Build command exited with code 1
```

**Solution:**
- Test build command locally
- Check file paths and permissions
- Verify all build dependencies

### 10. API Endpoint Issues

#### Issue: 404 Not Found
```
Error: Endpoint not found
```

**Solution:**
- Verify route definitions
- Check URL patterns
- Ensure proper FastAPI router setup

#### Issue: CORS Errors
```
Error: CORS policy blocked request
```

**Solution:**
- Configure CORS middleware properly
- Add allowed origins for frontend
- Check preflight request handling

## Debugging Steps

### 1. Local Testing
```bash
# Test locally first
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Check environment variables
python check_env.py

# Test database connections
curl http://localhost:8000/db-health
curl http://localhost:8000/redis-health
```

### 2. Render Logs Analysis
1. Go to Render Dashboard
2. Select your service
3. Click on "Logs" tab
4. Look for error messages and stack traces

### 3. Health Check Endpoints
```bash
# Test health endpoints
curl https://your-app.onrender.com/health
curl https://your-app.onrender.com/db-health
curl https://your-app.onrender.com/redis-health
```

### 4. Environment Variables Check
```bash
# In Render shell (if available)
echo $MONGO_URI
echo $REDIS_HOST
printenv | grep -E "(MONGO|REDIS|API_KEY)"
```

## Prevention Best Practices

### 1. Pre-deployment Checklist
- [ ] Test application locally
- [ ] Verify all environment variables
- [ ] Test database connections
- [ ] Check requirements.txt
- [ ] Validate render.yaml
- [ ] Test API endpoints

### 2. Monitoring Setup
- [ ] Configure health check endpoints
- [ ] Set up logging
- [ ] Monitor response times
- [ ] Track error rates
- [ ] Set up alerts

### 3. Backup Strategy
- [ ] Database backups
- [ ] Environment variables backup
- [ ] Code repository backup
- [ ] Configuration files backup

## Getting Help

### 1. Render Support
- Check Render documentation
- Contact Render support
- Community forums

### 2. Application Logs
- Enable detailed logging
- Use structured logging
- Monitor error patterns

### 3. Database Provider Support
- MongoDB Atlas support
- Redis Cloud support
- Check provider status pages

## Emergency Recovery

### 1. Rollback Deployment
1. Go to Render Dashboard
2. Select service
3. Go to "Deploys" tab
4. Click "Rollback" on previous working version

### 2. Quick Fixes
- Restart service
- Clear build cache
- Update environment variables
- Check external service status

### 3. Data Recovery
- Restore from database backup
- Check data integrity
- Verify application functionality