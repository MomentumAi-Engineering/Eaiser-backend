# 🚀 Render Deployment Guide - Eaiser Backend

## Prerequisites
- ✅ GitHub repository: `MomentumAi-Engineering/Eaiser-backend`
- ✅ Render.yaml configuration file ready
- ✅ All dependencies in requirements.txt

## Step-by-Step Deployment Process

### 1. Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up/Login with GitHub account
3. Authorize Render to access your repositories

### 2. Create New Web Service
1. Click **"New +"** button in dashboard
2. Select **"Web Service"**
3. Choose **"Build and deploy from a Git repository"**

### 3. Connect Repository
1. Select **GitHub** as source
2. Search for: `MomentumAi-Engineering/Eaiser-backend`
3. Click **"Connect"**
4. Install Render GitHub App if prompted

### 4. Service Configuration
```
Name: eaiser-backend
Region: Oregon (US West) 
Branch: main
Root Directory: (leave blank)
Runtime: Python 3
Build Command: pip install --upgrade pip && pip install -r requirements.txt
Start Command: cd app && python -m uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
```

### 5. Environment Variables (CRITICAL) 🔑
Add these environment variables in Render dashboard:

#### Required API Keys:
- `SENDGRID_API_KEY` = Your SendGrid API key
- `OPENAI_API_KEY` = Your OpenAI API key  
- `GEMINI_API_KEY` = Your Google Gemini API key
- `GOOGLE_API_KEY` = Your Google API key

#### Database Configuration:
- `MONGO_URI` = mongodb+srv://snapfix:Chrishabh100@snapfixcluster.7po8ntf.mongodb.net/snapfix?retryWrites=true&w=majority&appName=SnapFixCluster&tls=true&tlsAllowInvalidCertificates=false

#### Application Settings:
- `ENV` = production
- `ENVIRONMENT` = production
- `PYTHON_VERSION` = 3.11.9
- `EMAIL_USER` = no-reply@eaiser.ai
- `LOG_LEVEL` = INFO

#### Performance Settings:
- `MAX_WORKERS` = 4
- `WORKER_CONNECTIONS` = 1000
- `CACHE_TTL` = 3600
- `RATE_LIMIT_PER_MINUTE` = 1000

### 6. Advanced Settings
- **Health Check Path**: `/health`
- **Auto Deploy**: Enable (recommended)
- **Plan**: Free (for testing) or Starter ($7/month)

### 7. Deploy Process
1. Click **"Create Web Service"**
2. Wait for build process (5-10 minutes)
3. Monitor logs for any errors
4. Test deployment URL once ready

### 8. Post-Deployment Verification
1. Check health endpoint: `https://your-app.onrender.com/health`
2. Test API endpoints
3. Verify database connectivity
4. Check email functionality

## Important Notes

### Security Best Practices:
- ✅ Never commit API keys to repository
- ✅ Use environment variables for all secrets
- ✅ Enable HTTPS (automatic on Render)
- ✅ Set proper CORS policies

### Performance Optimization:
- ✅ Use appropriate worker count (4 for Starter plan)
- ✅ Enable caching where possible
- ✅ Monitor response times
- ✅ Set up proper logging

### Troubleshooting Common Issues:

#### Build Failures:
- Check requirements.txt for conflicting versions
- Verify Python version compatibility
- Review build logs for specific errors

#### Runtime Errors:
- Check environment variables are set correctly
- Verify database connection strings
- Monitor application logs

#### Performance Issues:
- Adjust worker count based on plan
- Optimize database queries
- Implement proper caching

## Monitoring & Maintenance

### Health Monitoring:
- Use `/health` endpoint for uptime monitoring
- Set up alerts for service downtime
- Monitor response times and error rates

### Scaling Considerations:
- Free plan: Limited resources, sleeps after 15 min inactivity
- Starter plan: Always-on, better performance
- Pro plan: Auto-scaling, advanced features

## Support Resources
- [Render Documentation](https://render.com/docs)
- [Python Deployment Guide](https://render.com/docs/deploy-fastapi)
- [Environment Variables](https://render.com/docs/environment-variables)

---
**Deployment Status**: Ready for production ✅
**Last Updated**: January 2025