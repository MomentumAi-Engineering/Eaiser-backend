const autocannon = require('autocannon');
const fs = require('fs');
const path = require('path');

// Configuration for comprehensive load testing
const CONFIG = {
    BASE_URL: 'http://127.0.0.1:10000',
    DURATION: 60,  // 1 minute test
    CONNECTIONS: 200,  // High concurrent connections for startup load
    WARMUP_DURATION: 10,  // Warmup period
    RESULTS_DIR: './load-test-results'
};

// Ensure results directory exists
if (!fs.existsSync(CONFIG.RESULTS_DIR)) {
    fs.mkdirSync(CONFIG.RESULTS_DIR, { recursive: true });
}

// Test scenarios for different endpoints
const TEST_SCENARIOS = [
    {
        name: 'Health Check Endpoint',
        url: `${CONFIG.BASE_URL}/health`,
        method: 'GET',
        description: 'Basic health check - should handle high load'
    },
    {
        name: 'Issues Health Check',
        url: `${CONFIG.BASE_URL}/api/issues/health`,
        method: 'GET',
        description: 'Issues service health check with database ping'
    },
    {
        name: 'Get Issues List',
        url: `${CONFIG.BASE_URL}/api/issues`,
        method: 'GET',
        description: 'List all issues - database read operation'
    }
];

// Function to run a single load test
function runLoadTest(scenario, connections = CONFIG.CONNECTIONS, duration = CONFIG.DURATION) {
    return new Promise((resolve, reject) => {
        console.log(`\n🚀 Running load test: ${scenario.name}`);
        console.log(`📊 URL: ${scenario.url}`);
        console.log(`⚡ Connections: ${connections}, Duration: ${duration}s`);
        console.log(`📝 Description: ${scenario.description}`);
        console.log('─'.repeat(60));

        const instance = autocannon({
            url: scenario.url,
            duration: duration,
            connections: connections,
            method: scenario.method,
            headers: {
                'User-Agent': 'SnapFix-LoadTest/1.0'
            }
        }, (err, result) => {
            if (err) {
                console.error(`❌ Error in ${scenario.name}:`, err);
                reject(err);
                return;
            }

            // Calculate performance metrics
            const metrics = {
                scenario: scenario.name,
                url: scenario.url,
                duration: result.duration,
                connections: connections,
                requests: {
                    total: result.requests.total,
                    average: result.requests.average,
                    min: result.requests.min,
                    max: result.requests.max,
                    p99: result.requests.p99,
                    p95: result.requests.p95,
                    p90: result.requests.p90
                },
                latency: {
                    average: result.latency.average,
                    min: result.latency.min,
                    max: result.latency.max,
                    p99: result.latency.p99,
                    p95: result.latency.p95,
                    p90: result.latency.p90
                },
                throughput: {
                    average: result.throughput.average,
                    min: result.throughput.min,
                    max: result.throughput.max
                },
                errors: result.errors,
                timeouts: result.timeouts,
                non2xx: result.non2xx,
                bytes: result.bytes,
                timestamp: new Date().toISOString()
            };

            // Display results
            console.log(`\n📈 Results for ${scenario.name}:`);
            console.log(`✅ Total Requests: ${metrics.requests.total.toLocaleString()}`);
            console.log(`⚡ Requests/sec: ${metrics.requests.average.toFixed(2)}`);
            console.log(`🕐 Avg Latency: ${metrics.latency.average.toFixed(2)}ms`);
            console.log(`📊 P99 Latency: ${metrics.latency.p99.toFixed(2)}ms`);
            console.log(`💾 Throughput: ${(metrics.throughput.average / 1024 / 1024).toFixed(2)} MB/s`);
            console.log(`❌ Errors: ${metrics.errors}`);
            console.log(`⏰ Timeouts: ${metrics.timeouts}`);
            
            // Performance assessment
            const performanceGrade = assessPerformance(metrics);
            console.log(`🎯 Performance Grade: ${performanceGrade.grade} - ${performanceGrade.message}`);

            resolve(metrics);
        });
    });
}

// Function to assess performance and provide recommendations
function assessPerformance(metrics) {
    const rps = metrics.requests.average;
    const avgLatency = metrics.latency.average;
    const p99Latency = metrics.latency.p99;
    const errorRate = (metrics.errors + metrics.timeouts) / metrics.requests.total * 100;

    // Performance grading based on startup requirements
    if (rps >= 2000 && avgLatency <= 50 && p99Latency <= 200 && errorRate <= 1) {
        return { grade: 'A+ (Excellent)', message: 'Ready for 1 lakh+ traffic!' };
    } else if (rps >= 1500 && avgLatency <= 100 && p99Latency <= 500 && errorRate <= 2) {
        return { grade: 'A (Very Good)', message: 'Can handle high traffic with minor optimizations' };
    } else if (rps >= 1000 && avgLatency <= 200 && p99Latency <= 1000 && errorRate <= 5) {
        return { grade: 'B (Good)', message: 'Suitable for medium traffic, needs optimization for scale' };
    } else if (rps >= 500 && avgLatency <= 500 && errorRate <= 10) {
        return { grade: 'C (Fair)', message: 'Basic performance, significant optimization needed' };
    } else {
        return { grade: 'D (Needs Improvement)', message: 'Performance issues detected, immediate optimization required' };
    }
}

// Function to generate comprehensive report
function generateReport(allResults) {
    const report = {
        testSuite: 'SnapFix Comprehensive Load Test',
        timestamp: new Date().toISOString(),
        configuration: CONFIG,
        results: allResults,
        summary: {
            totalTests: allResults.length,
            averageRPS: allResults.reduce((sum, r) => sum + r.requests.average, 0) / allResults.length,
            averageLatency: allResults.reduce((sum, r) => sum + r.latency.average, 0) / allResults.length,
            totalErrors: allResults.reduce((sum, r) => sum + r.errors + r.timeouts, 0),
            recommendations: []
        }
    };

    // Add recommendations based on results
    const avgRPS = report.summary.averageRPS;
    const avgLatency = report.summary.averageLatency;
    
    if (avgRPS < 1000) {
        report.summary.recommendations.push('Consider implementing connection pooling and database indexing');
    }
    if (avgLatency > 100) {
        report.summary.recommendations.push('Optimize database queries and add caching layer (Redis)');
    }
    if (report.summary.totalErrors > 0) {
        report.summary.recommendations.push('Investigate and fix error sources for production readiness');
    }
    
    report.summary.recommendations.push('Consider implementing rate limiting for production deployment');
    report.summary.recommendations.push('Add monitoring and alerting (Prometheus/Grafana)');
    
    return report;
}

// Main execution function
async function runComprehensiveLoadTest() {
    console.log('🎯 SnapFix Comprehensive Load Testing Suite');
    console.log('=' .repeat(60));
    console.log(`🚀 Testing for startup scale: 1 lakh+ traffic capability`);
    console.log(`⚙️  Configuration: ${CONFIG.CONNECTIONS} connections, ${CONFIG.DURATION}s duration`);
    
    const allResults = [];
    
    try {
        // Run warmup test first
        console.log('\n🔥 Running warmup test...');
        await runLoadTest(TEST_SCENARIOS[0], 50, CONFIG.WARMUP_DURATION);
        
        // Run all test scenarios
        for (const scenario of TEST_SCENARIOS) {
            const result = await runLoadTest(scenario);
            allResults.push(result);
            
            // Wait between tests to avoid overwhelming the server
            await new Promise(resolve => setTimeout(resolve, 5000));
        }
        
        // Generate and save comprehensive report
        const report = generateReport(allResults);
        const reportPath = path.join(CONFIG.RESULTS_DIR, `load-test-report-${Date.now()}.json`);
        fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
        
        // Display final summary
        console.log('\n' + '='.repeat(60));
        console.log('📊 COMPREHENSIVE LOAD TEST SUMMARY');
        console.log('='.repeat(60));
        console.log(`📈 Average RPS across all tests: ${report.summary.averageRPS.toFixed(2)}`);
        console.log(`🕐 Average Latency: ${report.summary.averageLatency.toFixed(2)}ms`);
        console.log(`❌ Total Errors: ${report.summary.totalErrors}`);
        console.log(`\n🎯 STARTUP READINESS ASSESSMENT:`);
        
        if (report.summary.averageRPS >= 1500) {
            console.log('✅ EXCELLENT: Your API can handle 1 lakh+ traffic!');
            console.log('🚀 Ready for production deployment with high confidence');
        } else if (report.summary.averageRPS >= 1000) {
            console.log('⚠️  GOOD: Can handle significant traffic with optimizations');
            console.log('🔧 Minor performance tuning recommended before scaling');
        } else {
            console.log('❌ NEEDS IMPROVEMENT: Requires optimization for high traffic');
            console.log('🛠️  Significant performance improvements needed');
        }
        
        console.log('\n💡 RECOMMENDATIONS:');
        report.summary.recommendations.forEach((rec, index) => {
            console.log(`${index + 1}. ${rec}`);
        });
        
        console.log(`\n📁 Detailed report saved: ${reportPath}`);
        console.log('✅ Load testing completed successfully!');
        
    } catch (error) {
        console.error('❌ Load testing failed:', error);
        process.exit(1);
    }
}

// Run the comprehensive load test
runComprehensiveLoadTest();