const autocannon = require('autocannon');
const fs = require('fs');

// Test different endpoints for load testing
const HEALTH_URL = 'http://127.0.0.1:8000/health';  // Main health endpoint
const ISSUES_HEALTH_URL = 'http://127.0.0.1:8000/api/issues/health';  // Issues health endpoint
const LIST_URL = 'http://127.0.0.1:8000/api/issues';
const DURATION = 30;    
const CONNECTIONS = 100;  // Increased connections for better load testing

console.log(`🚀 Running GET load test on health endpoint`);
console.log(`📊 Testing with ${CONNECTIONS} concurrent connections for ${DURATION} seconds`);

const instance = autocannon({
    url: HEALTH_URL,
    duration: DURATION,
    connections: CONNECTIONS,
    method: 'GET'
}, (err, result) => {
    if (err) {
        console.error('❌ Error:', err);
        process.exit(1);
    }

    console.log('\n✅ Test Finished\n');
    autocannon.printResult(result);

    fs.writeFileSync(
        'post-results.json',
        JSON.stringify(result, null, 2)
    );
    console.log('📁 Results saved to post-results.json');
});

autocannon.track(instance, { renderProgressBar: true, renderLatencyTable: true });
