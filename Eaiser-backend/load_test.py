"""
Load Testing Script for Eaiser AI Backend
Tests system performance under 1 lakh+ concurrent users
"""

import asyncio
import aiohttp
import time
import json
import statistics
from typing import List, Dict, Any
import logging
from datetime import datetime
import argparse
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('load_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LoadTester:
    """
    Advanced load testing class for high-concurrency testing
    """
    
    def __init__(self, base_url: str = "http://localhost:10000"):
        self.base_url = base_url
        self.results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'error_details': [],
            'start_time': None,
            'end_time': None
        }
        
    async def make_request(self, session: aiohttp.ClientSession, endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
        """Make a single HTTP request and record metrics"""
        start_time = time.time()
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                async with session.get(url) as response:
                    response_time = time.time() - start_time
                    content = await response.text()
                    
                    return {
                        'success': True,
                        'status_code': response.status,
                        'response_time': response_time,
                        'content_length': len(content)
                    }
            
            elif method.upper() == "POST":
                async with session.post(url, json=data) as response:
                    response_time = time.time() - start_time
                    content = await response.text()
                    
                    return {
                        'success': True,
                        'status_code': response.status,
                        'response_time': response_time,
                        'content_length': len(content)
                    }
                    
        except Exception as e:
            response_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'response_time': response_time
            }
    
    async def health_check_test(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test health check endpoint"""
        return await self.make_request(session, "/health")
    
    async def api_endpoint_test(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test main API endpoints"""
        return await self.make_request(session, "/")
    
    async def auth_test(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test authentication endpoints"""
        test_data = {
            "username": f"test_user_{int(time.time())}",
            "password": "test_password_123"
        }
        return await self.make_request(session, "/auth/login", "POST", test_data)
    
    async def concurrent_user_simulation(self, num_users: int, duration_seconds: int = 60):
        """Simulate concurrent users for specified duration"""
        logger.info(f"🚀 Starting load test with {num_users} concurrent users for {duration_seconds} seconds")
        
        self.results['start_time'] = datetime.now()
        
        # Create connector with optimized settings
        connector = aiohttp.TCPConnector(
            limit=num_users * 2,  # Connection pool size
            limit_per_host=num_users,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        # Create session with timeout settings
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'LoadTester/1.0'}
        ) as session:
            
            # Create tasks for concurrent users
            tasks = []
            
            for user_id in range(num_users):
                task = asyncio.create_task(
                    self.simulate_user_session(session, user_id, duration_seconds)
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed: {result}")
                    self.results['failed_requests'] += 1
                else:
                    self.process_user_results(result)
        
        self.results['end_time'] = datetime.now()
        return self.generate_report()
    
    async def simulate_user_session(self, session: aiohttp.ClientSession, user_id: int, duration_seconds: int) -> List[Dict[str, Any]]:
        """Simulate a single user's session"""
        user_results = []
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time:
            try:
                # Simulate different user behaviors
                actions = [
                    self.health_check_test,
                    self.api_endpoint_test,
                    # self.auth_test  # Uncomment when auth endpoints are ready
                ]
                
                # Random action selection
                import random
                action = random.choice(actions)
                result = await action(session)
                user_results.append(result)
                
                # Random delay between requests (0.1 to 2 seconds)
                await asyncio.sleep(random.uniform(0.1, 2.0))
                
            except Exception as e:
                logger.error(f"User {user_id} error: {e}")
                user_results.append({
                    'success': False,
                    'error': str(e),
                    'response_time': 0
                })
        
        return user_results
    
    def process_user_results(self, user_results: List[Dict[str, Any]]):
        """Process results from a single user session"""
        for result in user_results:
            self.results['total_requests'] += 1
            
            if result['success']:
                self.results['successful_requests'] += 1
                self.results['response_times'].append(result['response_time'])
            else:
                self.results['failed_requests'] += 1
                self.results['error_details'].append(result.get('error', 'Unknown error'))
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.results['response_times']:
            return {
                'error': 'No successful requests to analyze',
                'total_requests': self.results['total_requests'],
                'failed_requests': self.results['failed_requests']
            }
        
        # Calculate statistics
        response_times = self.results['response_times']
        total_duration = (self.results['end_time'] - self.results['start_time']).total_seconds()
        
        report = {
            'test_summary': {
                'total_requests': self.results['total_requests'],
                'successful_requests': self.results['successful_requests'],
                'failed_requests': self.results['failed_requests'],
                'success_rate': (self.results['successful_requests'] / max(self.results['total_requests'], 1)) * 100,
                'test_duration': total_duration,
                'requests_per_second': self.results['total_requests'] / max(total_duration, 1)
            },
            'performance_metrics': {
                'avg_response_time': statistics.mean(response_times),
                'median_response_time': statistics.median(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times),
                'p95_response_time': self.percentile(response_times, 95),
                'p99_response_time': self.percentile(response_times, 99)
            },
            'error_analysis': {
                'total_errors': len(self.results['error_details']),
                'unique_errors': len(set(self.results['error_details'])),
                'error_samples': self.results['error_details'][:10]  # First 10 errors
            }
        }
        
        return report
    
    @staticmethod
    def percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile of response times"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

async def run_progressive_load_test():
    """Run progressive load test with increasing user counts"""
    tester = LoadTester()
    
    # Test scenarios: (users, duration)
    test_scenarios = [
        (100, 30),      # 100 users for 30 seconds
        (500, 30),      # 500 users for 30 seconds
        (1000, 60),     # 1K users for 1 minute
        (5000, 60),     # 5K users for 1 minute
        (10000, 120),   # 10K users for 2 minutes
        (50000, 180),   # 50K users for 3 minutes
        (100000, 300)   # 100K users for 5 minutes
    ]
    
    all_results = {}
    
    for users, duration in test_scenarios:
        logger.info(f"🔥 Starting test: {users} users for {duration} seconds")
        
        try:
            result = await tester.concurrent_user_simulation(users, duration)
            all_results[f"{users}_users"] = result
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"TEST RESULTS: {users} Users")
            print(f"{'='*60}")
            print(f"Success Rate: {result['test_summary']['success_rate']:.2f}%")
            print(f"Requests/sec: {result['test_summary']['requests_per_second']:.2f}")
            print(f"Avg Response: {result['performance_metrics']['avg_response_time']:.3f}s")
            print(f"P95 Response: {result['performance_metrics']['p95_response_time']:.3f}s")
            print(f"P99 Response: {result['performance_metrics']['p99_response_time']:.3f}s")
            
            # Wait between tests
            if users < 100000:
                logger.info("⏳ Waiting 30 seconds before next test...")
                await asyncio.sleep(30)
                
        except Exception as e:
            logger.error(f"❌ Test failed for {users} users: {e}")
            all_results[f"{users}_users"] = {'error': str(e)}
    
    # Save detailed results
    with open('load_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info("📊 Load test completed! Results saved to load_test_results.json")
    return all_results

async def quick_health_test():
    """Quick health check test"""
    tester = LoadTester()
    
    logger.info("🏥 Running quick health check...")
    
    connector = aiohttp.TCPConnector(limit=10)
    timeout = aiohttp.ClientTimeout(total=10)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        result = await tester.health_check_test(session)
        
        if result['success']:
            logger.info(f"✅ Health check passed - Response time: {result['response_time']:.3f}s")
            return True
        else:
            logger.error(f"❌ Health check failed: {result.get('error', 'Unknown error')}")
            return False

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Load test Eaiser AI Backend')
    parser.add_argument('--url', default='http://localhost:10000', help='Base URL to test')
    parser.add_argument('--users', type=int, default=1000, help='Number of concurrent users')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--quick', action='store_true', help='Run quick health check only')
    parser.add_argument('--progressive', action='store_true', help='Run progressive load test')
    
    args = parser.parse_args()
    
    # Set base URL
    tester = LoadTester(args.url)
    
    if args.quick:
        # Quick health check
        result = asyncio.run(quick_health_test())
        sys.exit(0 if result else 1)
    
    elif args.progressive:
        # Progressive load test
        asyncio.run(run_progressive_load_test())
    
    else:
        # Single load test
        async def single_test():
            result = await tester.concurrent_user_simulation(args.users, args.duration)
            print(json.dumps(result, indent=2, default=str))
        
        asyncio.run(single_test())

if __name__ == "__main__":
    main()