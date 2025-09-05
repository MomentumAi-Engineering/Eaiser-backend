
// Performance monitoring script
const { execSync } = require('child_process');
const fs = require('fs');

function collectMetrics() {
  const timestamp = new Date().toISOString();
  
  try {
    // Get Docker stats
    const dockerStats = execSync('docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"', { encoding: 'utf8' });
    
    // Get system metrics
    const systemMetrics = {
      timestamp,
      docker_stats: dockerStats,
      memory_usage: process.memoryUsage(),
      cpu_usage: process.cpuUsage()
    };
    
    // Append to metrics file
    fs.appendFileSync('./load-test-metrics.json', JSON.stringify(systemMetrics) + '
');
    
  } catch (error) {
    console.error('Error collecting metrics:', error.message);
  }
}

// Collect metrics every 10 seconds
setInterval(collectMetrics, 10000);
console.log('📊 Monitoring started - metrics saved to load-test-metrics.json');
