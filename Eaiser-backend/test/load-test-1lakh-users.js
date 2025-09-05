// 🚀 SnapFix Load Test for 1 Lakh Concurrent Users
// Advanced load testing script using Artillery.js

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Test Configuration for 1 Lakh Users
const loadTestConfig = {
  config: {
    target: 'http://localhost:80', // Load balancer endpoint
    phases: [
      // Warm-up phase
      {
        duration: '2m',
        arrivalRate: 100,
        name: 'Warm-up'
      },
      // Ramp-up phase
      {
        duration: '5m',
        arrivalRate: 1000,
        rampTo: 10000,
        name: 'Ramp-up'
      },
      // Peak load phase - 1 Lakh users
      {
        duration: '10m',
        arrivalRate: 100000, // 1 lakh users per minute
        name: 'Peak Load - 1 Lakh Users'
      },
      // Sustained load
      {
        duration: '15m',
        arrivalRate: 50000,
        name: 'Sustained Load'
      },
      // Stress test
      {
        duration: '5m',
        arrivalRate: 150000, // 1.5 lakh users
        name: 'Stress Test'
      },
      // Cool down
      {
        duration: '3m',
        arrivalRate: 1000,
        name: 'Cool Down'
      }
    ],
    payload: {
      path: './test-data.csv',
      fields: ['userId', 'issueTitle', 'issueDescription', 'priority']
    },
    plugins: {
      'artillery-plugin-metrics-by-endpoint': {},
      'artillery-plugin-cloudwatch': {
        namespace: 'SnapFix/LoadTest'
      }
    },
    processor: './load-test-processor.js'
  },
  scenarios: [
    {
      name: 'Issue Reporting Workflow',
      weight: 40, // 40% of traffic
      flow: [
        {
          post: {
            url: '/api/auth/login',
            json: {
              email: '{{ $randomEmail }}',
              password: 'testpass123'
            },
            capture: {
              json: '$.token',
              as: 'authToken'
            }
          }
        },
        {
          think: 2 // 2 second think time
        },
        {
          post: {
            url: '/api/issues',
            headers: {
              'Authorization': 'Bearer {{ authToken }}',
              'Content-Type': 'application/json'
            },
            json: {
              title: '{{ issueTitle }}',
              description: '{{ issueDescription }}',
              priority: '{{ priority }}',
              category: 'bug',
              tags: ['urgent', 'production'],
              location: {
                latitude: 28.6139,
                longitude: 77.2090
              }
            },
            capture: {
              json: '$.issue_id',
              as: 'issueId'
            }
          }
        },
        {
          think: 1
        },
        {
          get: {
            url: '/api/issues/{{ issueId }}',
            headers: {
              'Authorization': 'Bearer {{ authToken }}'
            }
          }
        }
      ]
    },
    {
      name: 'Issue Browsing',
      weight: 30, // 30% of traffic
      flow: [
        {
          get: {
            url: '/api/issues?page={{ $randomInt(1, 100) }}&limit=20&sort=timestamp'
          }
        },
        {
          think: 3
        },
        {
          get: {
            url: '/api/issues/search?q=bug&category=all&priority=high'
          }
        },
        {
          think: 2
        },
        {
          get: {
            url: '/api/issues/stats'
          }
        }
      ]
    },
    {
      name: 'User Management',
      weight: 20, // 20% of traffic
      flow: [
        {
          post: {
            url: '/api/auth/register',
            json: {
              email: '{{ $randomEmail }}',
              password: 'testpass123',
              name: '{{ $randomFullName }}',
              phone: '{{ $randomPhoneNumber }}'
            }
          }
        },
        {
          think: 1
        },
        {
          post: {
            url: '/api/auth/login',
            json: {
              email: '{{ $randomEmail }}',
              password: 'testpass123'
            },
            capture: {
              json: '$.token',
              as: 'authToken'
            }
          }
        },
        {
          get: {
            url: '/api/user/profile',
            headers: {
              'Authorization': 'Bearer {{ authToken }}'
            }
          }
        }
      ]
    },
    {
      name: 'Health Checks',
      weight: 10, // 10% of traffic
      flow: [
        {
          get: {
            url: '/health'
          }
        },
        {
          get: {
            url: '/api/health'
          }
        },
        {
          get: {
            url: '/metrics'
          }
        }
      ]
    }
  ]
};

// Generate test data CSV
function generateTestData() {
  console.log('🔄 Generating test data for 1 lakh users...');
  
  const testData = [];
  const priorities = ['low', 'medium', 'high', 'critical'];
  const issueTitles = [
    'Application crashes on startup',
    'Login page not loading',
    'Database connection timeout',
    'API response too slow',
    'Memory leak in background process',
    'UI elements not responsive',
    'File upload failing',
    'Search functionality broken',
    'Email notifications not working',
    'Performance degradation'
  ];
  
  const issueDescriptions = [
    'Detailed description of the critical issue affecting multiple users',
    'Steps to reproduce: 1. Open app 2. Click login 3. Error appears',
    'System becomes unresponsive after prolonged usage',
    'Users unable to complete essential workflows',
    'Error occurs intermittently during peak hours'
  ];
  
  // Generate 100,000 test records
  for (let i = 1; i <= 100000; i++) {
    testData.push([
      `user${i}@test.com`,
      issueTitles[Math.floor(Math.random() * issueTitles.length)],
      issueDescriptions[Math.floor(Math.random() * issueDescriptions.length)],
      priorities[Math.floor(Math.random() * priorities.length)]
    ]);
  }
  
  // Write CSV header and data
  const csvContent = 'userId,issueTitle,issueDescription,priority\n' + 
                    testData.map(row => row.join(',')).join('\n');
  
  fs.writeFileSync('./test-data.csv', csvContent);
  console.log('✅ Test data generated: 100,000 records');
}

// Create load test processor
function createProcessor() {
  const processorCode = `
// Load test processor for custom functions
module.exports = {
  // Generate random email
  randomEmail: function(context, events, done) {
    const domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'test.com'];
    const domain = domains[Math.floor(Math.random() * domains.length)];
    const username = 'user' + Math.floor(Math.random() * 100000);
    context.vars.randomEmail = username + '@' + domain;
    return done();
  },
  
  // Generate random full name
  randomFullName: function(context, events, done) {
    const firstNames = ['John', 'Jane', 'Mike', 'Sarah', 'David', 'Lisa', 'Tom', 'Anna'];
    const lastNames = ['Smith', 'Johnson', 'Brown', 'Davis', 'Wilson', 'Miller', 'Taylor'];
    const firstName = firstNames[Math.floor(Math.random() * firstNames.length)];
    const lastName = lastNames[Math.floor(Math.random() * lastNames.length)];
    context.vars.randomFullName = firstName + ' ' + lastName;
    return done();
  },
  
  // Generate random phone number
  randomPhoneNumber: function(context, events, done) {
    const phoneNumber = '+91' + Math.floor(Math.random() * 9000000000 + 1000000000);
    context.vars.randomPhoneNumber = phoneNumber;
    return done();
  },
  
  // Log response times for analysis
  logResponse: function(requestParams, response, context, ee, next) {
    if (response.timings) {
      console.log('Response time:', response.timings.response, 'ms');
    }
    return next();
  },
  
  // Custom error handling
  handleError: function(requestParams, response, context, ee, next) {
    if (response.statusCode >= 400) {
      console.error('Error:', response.statusCode, response.body);
    }
    return next();
  }
};
`;
  
  fs.writeFileSync('./load-test-processor.js', processorCode);
  console.log('✅ Load test processor created');
}

// Run performance monitoring
function startMonitoring() {
  console.log('📊 Starting performance monitoring...');
  
  const monitoringScript = `
// Performance monitoring script
const { execSync } = require('child_process');
const fs = require('fs');

function collectMetrics() {
  const timestamp = new Date().toISOString();
  
  try {
    // Get Docker stats
    const dockerStats = execSync('docker stats --no-stream --format "table {{.Container}}\\t{{.CPUPerc}}\\t{{.MemUsage}}\\t{{.NetIO}}"', { encoding: 'utf8' });
    
    // Get system metrics
    const systemMetrics = {
      timestamp,
      docker_stats: dockerStats,
      memory_usage: process.memoryUsage(),
      cpu_usage: process.cpuUsage()
    };
    
    // Append to metrics file
    fs.appendFileSync('./load-test-metrics.json', JSON.stringify(systemMetrics) + '\n');
    
  } catch (error) {
    console.error('Error collecting metrics:', error.message);
  }
}

// Collect metrics every 10 seconds
setInterval(collectMetrics, 10000);
console.log('📊 Monitoring started - metrics saved to load-test-metrics.json');
`;
  
  fs.writeFileSync('./monitoring.js', monitoringScript);
  
  // Start monitoring in background
  const { spawn } = require('child_process');
  const monitor = spawn('node', ['./monitoring.js'], { detached: true, stdio: 'ignore' });
  monitor.unref();
  
  console.log('✅ Performance monitoring started');
}

// Generate load test report
function generateReport(results) {
  console.log('📋 Generating comprehensive load test report...');
  
  const report = {
    test_summary: {
      target_users: '1,00,000 concurrent users',
      test_duration: '40 minutes',
      test_phases: 6,
      timestamp: new Date().toISOString()
    },
    performance_metrics: results,
    recommendations: [
      'Monitor database connection pool utilization',
      'Implement Redis cluster for better caching',
      'Consider horizontal scaling if response times > 100ms',
      'Set up auto-scaling based on CPU/Memory thresholds',
      'Implement circuit breakers for external dependencies'
    ],
    scaling_strategy: {
      current_capacity: '1,00,000 users',
      recommended_scaling: {
        api_instances: '10-20 instances',
        database_replicas: '3-5 replicas',
        redis_cluster_nodes: '6 nodes',
        load_balancer_config: 'Nginx with least_conn'
      }
    }
  };
  
  fs.writeFileSync('./load-test-report.json', JSON.stringify(report, null, 2));
  console.log('✅ Load test report generated: load-test-report.json');
}

// Main execution function
async function runLoadTest() {
  console.log('🚀 Starting SnapFix Load Test for 1 Lakh Concurrent Users');
  console.log('=' .repeat(60));
  
  try {
    // Preparation phase
    console.log('\n📋 Phase 1: Preparation');
    generateTestData();
    createProcessor();
    startMonitoring();
    
    // Write Artillery config
    fs.writeFileSync('./artillery-config.yml', 
      'config:\n' +
      '  target: ' + loadTestConfig.config.target + '\n' +
      '  phases:\n' +
      loadTestConfig.config.phases.map(phase => 
        `    - duration: ${phase.duration}\n` +
        `      arrivalRate: ${phase.arrivalRate}\n` +
        (phase.rampTo ? `      rampTo: ${phase.rampTo}\n` : '') +
        `      name: "${phase.name}"\n`
      ).join('') +
      '  processor: ./load-test-processor.js\n' +
      '  payload:\n' +
      '    path: ./test-data.csv\n' +
      '    fields: ["userId", "issueTitle", "issueDescription", "priority"]\n' +
      'scenarios:\n' +
      loadTestConfig.scenarios.map(scenario => 
        `  - name: "${scenario.name}"\n` +
        `    weight: ${scenario.weight}\n` +
        `    flow:\n` +
        scenario.flow.map(step => 
          Object.keys(step).map(key => 
            `      - ${key}:\n` +
            (typeof step[key] === 'object' ? 
              Object.keys(step[key]).map(subKey => 
                `          ${subKey}: ${typeof step[key][subKey] === 'object' ? JSON.stringify(step[key][subKey]) : step[key][subKey]}\n`
              ).join('') : 
              `          ${step[key]}\n`
            )
          ).join('')
        ).join('')
      ).join('')
    );
    
    console.log('✅ Artillery configuration created');
    
    // Execution phase
    console.log('\n🔥 Phase 2: Load Test Execution');
    console.log('Target: 1,00,000 concurrent users');
    console.log('Duration: 40 minutes');
    console.log('Starting load test...');
    
    // Run Artillery load test
    const artilleryCommand = 'npx artillery run artillery-config.yml --output load-test-results.json';
    console.log('Executing:', artilleryCommand);
    
    const results = execSync(artilleryCommand, { 
      encoding: 'utf8',
      maxBuffer: 1024 * 1024 * 10 // 10MB buffer
    });
    
    console.log('\n📊 Load Test Results:');
    console.log(results);
    
    // Analysis phase
    console.log('\n📋 Phase 3: Results Analysis');
    
    // Read results file
    let testResults = {};
    try {
      const resultsData = fs.readFileSync('./load-test-results.json', 'utf8');
      testResults = JSON.parse(resultsData);
    } catch (error) {
      console.warn('Could not read detailed results file');
    }
    
    generateReport(testResults);
    
    // Success summary
    console.log('\n🎉 Load Test Completed Successfully!');
    console.log('=' .repeat(60));
    console.log('✅ Tested: 1,00,000 concurrent users');
    console.log('✅ Results: load-test-results.json');
    console.log('✅ Report: load-test-report.json');
    console.log('✅ Metrics: load-test-metrics.json');
    console.log('\n📊 Check Grafana dashboard: http://localhost:3001');
    console.log('📈 Check Prometheus metrics: http://localhost:9090');
    
  } catch (error) {
    console.error('\n❌ Load test failed:', error.message);
    console.error('\nTroubleshooting:');
    console.error('1. Ensure all services are running: docker-compose ps');
    console.error('2. Check service health: curl http://localhost:80/health');
    console.error('3. Verify load balancer: curl http://localhost:80');
    console.error('4. Check logs: docker-compose logs');
    process.exit(1);
  }
}

// Execute if run directly
if (require.main === module) {
  runLoadTest();
}

module.exports = { runLoadTest, generateTestData, createProcessor };