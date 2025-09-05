const autocannon = require('autocannon');

// Config
const TARGET_URL = 'http://127.0.0.1:8000/api/issues'; // Local backend URL
const CONNECTIONS = 100; // concurrent users
const DURATION = 30; // seconds

console.log(`🚀 Starting load test: ${CONNECTIONS} concurrent users for ${DURATION} seconds...`);

const instance = autocannon({
    url: TARGET_URL,
    connections: CONNECTIONS,
    duration: DURATION,
    headers: {
        'Content-Type': 'application/json'
    },
    // Example body if POST request (optional)
    // method: 'POST',
    // body: JSON.stringify({ key: 'value' }),
}, (err, result) => {
    if (err) {
        console.error('❌ Error:', err);
        process.exit(1);
    }
    console.log('\n✅ Test Finished\n');
    autocannon.printResult(result);
});

// Live progress
autocannon.track(instance, { renderProgressBar: true, renderLatencyTable: true });
