# Production Setup Guide for Eaiser Backend

## Redis Configuration for Production

### Step 1: Setup Upstash Redis (Free Tier)
1. Go to [Upstash Console](https://console.upstash.com/)
2. Create a new Redis database
3. Copy the Redis URL (format: `redis://default:password@host:port`)

### Step 2: Configure Render Environment Variables
In your Render dashboard, add these environment variables:

```bash
# Required: Redis URL from Upstash
REDIS_URL=redis://default:your_password@your_host:your_port

# AI Service Timeouts (already configured in render.yaml)
AI_TIMEOUT=15
GEMINI_TIMEOUT=15
```

## Performance Optimizations Applied

### 1. Increased Timeouts
- AI service timeout: 5s → 15s for production
- Gemini API timeout: 5s → 15s for production
- Keepalive timeout: 2s → 5s for stability

### 2. Enhanced Caching Strategy
- Redis enabled for production caching
- Fallback reports cached for 1 hour
- Location and timezone data cached

### 3. Better Error Handling
- Graceful timeout handling with structured fallback reports
- Improved logging with timeout values
- Production-ready error messages

## Deployment Steps

1. **Update Environment Variables in Render:**
   ```bash
   REDIS_URL=your_upstash_redis_url
   ```

2. **Redeploy the Service:**
   - Push changes to your Git repository
   - Render will auto-deploy with new configurations

3. **Verify Performance:**
   - Test report generation (should be <5 seconds)
   - Check Redis connectivity in logs
   - Monitor response times

## Expected Performance Improvements

- **Before:** 24 seconds (no caching, short timeouts)
- **After:** 2-5 seconds (with Redis caching, proper timeouts)

## Troubleshooting

### If still slow:
1. Check Redis connectivity in Render logs
2. Verify REDIS_URL is correctly set
3. Monitor Gemini API response times
4. Check network latency to Gemini API

### Logs to Monitor:
- Redis connection status
- AI service timeout logs
- Cache hit/miss rates
- Response time metrics