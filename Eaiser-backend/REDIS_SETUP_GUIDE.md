# Redis Setup Guide for Production

## Overview
Complete guide for setting up Redis for Eaiser Backend in production environment.

## Redis Cloud Setup (Recommended)

### 1. Create Redis Cloud Account
1. Go to [Redis Cloud](https://redis.com/redis-enterprise-cloud/)
2. Sign up for free tier or paid plan
3. Create new database instance

### 2. Configuration
```bash
# Redis Cloud provides these details
REDIS_HOST=redis-xxxxx.c1.us-east-1-1.ec2.cloud.redislabs.com
REDIS_PORT=6379
REDIS_PASSWORD=your-generated-password
REDIS_DB=0
REDIS_SSL=true
```

### 3. Security Settings
- Enable password authentication
- Use SSL/TLS encryption
- Configure IP whitelist if needed
- Set memory limits and eviction policies

## Self-Hosted Redis Setup

### 1. Installation (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install redis-server
```

### 2. Configuration File
Edit `/etc/redis/redis.conf`:
```bash
# Bind to specific IP
bind 127.0.0.1 your-server-ip

# Set password
requirepass your-strong-password

# Enable persistence
save 900 1
save 300 10
save 60 10000

# Set memory limit
maxmemory 256mb
maxmemory-policy allkeys-lru

# Enable SSL (optional)
tls-port 6380
tls-cert-file /path/to/redis.crt
tls-key-file /path/to/redis.key
```

### 3. Start Redis Service
```bash
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

## Application Integration

### 1. Environment Variables
```bash
REDIS_HOST=your-redis-host
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password
REDIS_DB=0
REDIS_SSL=true  # for production
```

### 2. Connection Testing
Use the health check endpoint:
```bash
curl http://your-app-url/redis-health
```

### 3. Redis Usage in Application
- Session storage
- API response caching
- Rate limiting data
- Temporary data storage

## Performance Tuning

### 1. Memory Configuration
```bash
# Set appropriate memory limit
maxmemory 512mb

# Choose eviction policy
maxmemory-policy allkeys-lru
```

### 2. Persistence Settings
```bash
# RDB snapshots
save 900 1
save 300 10
save 60 10000

# AOF (Append Only File)
appendonly yes
appendfsync everysec
```

### 3. Connection Pooling
Application uses connection pooling for optimal performance:
- Max connections: 20
- Connection timeout: 5 seconds
- Retry attempts: 3

## Monitoring

### 1. Redis CLI Commands
```bash
# Connect to Redis
redis-cli -h host -p port -a password

# Check memory usage
INFO memory

# Monitor commands
MONITOR

# Check connected clients
CLIENT LIST
```

### 2. Key Metrics to Monitor
- Memory usage
- Connected clients
- Commands per second
- Hit/miss ratio
- Persistence status

## Security Best Practices

### 1. Authentication
- Always use strong passwords
- Enable AUTH command
- Consider using ACL (Access Control Lists)

### 2. Network Security
- Use SSL/TLS encryption
- Configure firewall rules
- Bind to specific interfaces only
- Use VPN for remote access

### 3. Data Protection
- Enable persistence (RDB + AOF)
- Regular backups
- Monitor for unusual activity

## Troubleshooting

### 1. Connection Issues
```bash
# Test connection
redis-cli -h host -p port ping

# Check if Redis is running
sudo systemctl status redis-server

# View Redis logs
sudo journalctl -u redis-server
```

### 2. Memory Issues
```bash
# Check memory usage
redis-cli INFO memory

# Clear all data (use carefully)
redis-cli FLUSHALL
```

### 3. Performance Issues
```bash
# Monitor slow queries
redis-cli --latency
redis-cli --latency-history

# Check configuration
redis-cli CONFIG GET "*"
```

## Backup and Recovery

### 1. RDB Backup
```bash
# Manual backup
redis-cli BGSAVE

# Backup file location
/var/lib/redis/dump.rdb
```

### 2. AOF Backup
```bash
# AOF file location
/var/lib/redis/appendonly.aof

# Rewrite AOF
redis-cli BGREWRITEAOF
```

### 3. Recovery
1. Stop Redis service
2. Replace dump.rdb or appendonly.aof
3. Start Redis service
4. Verify data integrity

## Production Checklist

- [ ] Redis instance configured with authentication
- [ ] SSL/TLS encryption enabled
- [ ] Memory limits set appropriately
- [ ] Persistence configured (RDB + AOF)
- [ ] Monitoring setup
- [ ] Backup strategy implemented
- [ ] Security measures in place
- [ ] Application connection tested
- [ ] Performance benchmarks established
- [ ] Documentation updated