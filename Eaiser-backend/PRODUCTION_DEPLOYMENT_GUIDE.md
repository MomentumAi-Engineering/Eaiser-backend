# Production Deployment Guide - Eaiser Backend

## Overview
Complete guide for deploying Eaiser Backend to production environment with MongoDB Atlas and Redis.

## Prerequisites
- MongoDB Atlas cluster setup
- Redis instance (Redis Cloud or self-hosted)
- Render.com account (or preferred hosting platform)
- Environment variables configured

## Environment Variables Required

### Database Configuration
```bash
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority&ssl=true&tls=true
MONGODB_NAME=your-database-name
```

### Redis Configuration
```bash
REDIS_HOST=your-redis-host
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password
```

### AI Services
```bash
GOOGLE_API_KEY=your-google-api-key
GEMINI_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key
```

### Email Configuration
```bash
SENDGRID_API_KEY=your-sendgrid-api-key
EMAIL_USER=your-email@domain.com
```

## Deployment Steps

### 1. Environment Setup
1. Copy `.env.production.example` to `.env.production`
2. Fill in all required environment variables
3. Never commit `.env.production` to version control

### 2. Database Setup
- MongoDB Atlas cluster should be configured with proper security
- Create database indexes using `app/create_indexes.py`
- Ensure proper connection string with SSL/TLS

### 3. Redis Setup
- Configure Redis instance with password authentication
- Enable SSL/TLS for production
- Set appropriate memory limits and eviction policies

### 4. Render.com Deployment
1. Connect GitHub repository to Render
2. Use `render.yaml` configuration
3. Set environment variables in Render dashboard
4. Deploy and monitor logs

### 5. Health Checks
- `/health` - Basic health check
- `/db-health` - MongoDB connection check
- `/redis-health` - Redis connection check

## Performance Optimizations
- Gunicorn with multiple workers
- Connection pooling for MongoDB
- Redis caching for frequent queries
- Proper logging and monitoring

## Security Considerations
- All API keys stored as environment variables
- MongoDB connection with authentication
- Redis password protection
- HTTPS enforcement
- Rate limiting enabled

## Monitoring
- Application logs via Render
- Database performance monitoring
- Redis memory usage tracking
- API response time monitoring

## Troubleshooting
1. Check environment variables are set correctly
2. Verify database connection strings
3. Ensure Redis connectivity
4. Review application logs
5. Test API endpoints individually

## Support
For deployment issues, check:
- `RENDER_DEPLOYMENT_TROUBLESHOOTING.md`
- `REDIS_SETUP_GUIDE.md`
- Application logs in Render dashboard