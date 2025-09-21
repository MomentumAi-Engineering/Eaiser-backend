# 🚀 Report Generation Performance Test
# Test script to measure report generation speed and capacity

import asyncio
import aiohttp
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict, Any
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReportPerformanceTester:
    """
    High-performance report generation tester
    Measures system capability for generating reports at scale
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:10000"):
        self.base_url = base_url
        self.session = None
        self.results = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=50,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def test_single_report_speed(self, report_type: str = "performance") -> Dict[str, Any]:
        """Test single report generation speed"""
        logger.info(f"🚀 Testing single {report_type} report generation speed...")
        
        start_time = time.time()
        
        try:
            # Test data for report generation
            test_payload = {
                "report_type": report_type,
                "format": "json",
                "priority": 1,
                "cache_ttl": 300
            }
            
            async with self.session.post(
                f"{self.base_url}/api/reports/generate",
                json=test_payload
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    generation_time = time.time() - start_time
                    
                    return {
                        "success": True,
                        "generation_time": generation_time,
                        "server_time": result.get("generation_time", 0),
                        "cache_hit": result.get("cache_hit", False),
                        "size_bytes": result.get("size_bytes", 0),
                        "report_type": report_type
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Report generation failed: {response.status} - {error_text}")
                    return {"success": False, "error": f"HTTP {response.status}"}
                    
        except Exception as e:
            logger.error(f"❌ Exception in single report test: {str(e)}")
            return {"success": False, "error": str(e)}

    async def test_concurrent_reports(self, concurrent_count: int = 10, report_type: str = "performance") -> Dict[str, Any]:
        """Test concurrent report generation"""
        logger.info(f"⚡ Testing {concurrent_count} concurrent {report_type} reports...")
        
        start_time = time.time()
        
        # Create concurrent tasks
        tasks = []
        for i in range(concurrent_count):
            task = self.test_single_report_speed(report_type)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Process results
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed_results = [r for r in results if isinstance(r, Exception) or (isinstance(r, dict) and not r.get("success"))]
        
        if successful_results:
            generation_times = [r["generation_time"] for r in successful_results]
            server_times = [r["server_time"] for r in successful_results]
            cache_hits = sum(1 for r in successful_results if r.get("cache_hit"))
            
            return {
                "concurrent_count": concurrent_count,
                "successful": len(successful_results),
                "failed": len(failed_results),
                "total_time": total_time,
                "reports_per_second": len(successful_results) / total_time,
                "avg_generation_time": statistics.mean(generation_times),
                "median_generation_time": statistics.median(generation_times),
                "min_generation_time": min(generation_times),
                "max_generation_time": max(generation_times),
                "avg_server_time": statistics.mean(server_times),
                "cache_hit_rate": (cache_hits / len(successful_results)) * 100,
                "success_rate": (len(successful_results) / concurrent_count) * 100
            }
        else:
            return {
                "concurrent_count": concurrent_count,
                "successful": 0,
                "failed": len(failed_results),
                "total_time": total_time,
                "error": "All requests failed"
            }

    async def test_bulk_reports(self, bulk_size: int = 5) -> Dict[str, Any]:
        """Test bulk report generation"""
        logger.info(f"📊 Testing bulk report generation with {bulk_size} reports...")
        
        start_time = time.time()
        
        # Create bulk request payload
        bulk_payload = {
            "reports": [
                {"report_type": "performance", "format": "json", "priority": 1},
                {"report_type": "user_analytics", "format": "json", "priority": 1},
                {"report_type": "system_health", "format": "json", "priority": 1},
                {"report_type": "load_test", "format": "json", "priority": 2},
                {"report_type": "security", "format": "json", "priority": 2}
            ][:bulk_size]
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/reports/generate/bulk",
                json=bulk_payload
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    total_time = time.time() - start_time
                    
                    return {
                        "success": True,
                        "bulk_size": bulk_size,
                        "total_time": total_time,
                        "completed": result.get("completed", 0),
                        "failed": result.get("failed", 0),
                        "reports_per_second": bulk_size / total_time,
                        "results": result.get("results", [])
                    }
                else:
                    error_text = await response.text()
                    return {"success": False, "error": f"HTTP {response.status} - {error_text}"}
                    
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_different_report_types(self) -> Dict[str, Any]:
        """Test generation speed for different report types"""
        logger.info("📋 Testing different report types...")
        
        report_types = ["performance", "user_analytics", "system_health", "load_test", "security", "business"]
        results = {}
        
        for report_type in report_types:
            logger.info(f"Testing {report_type} report...")
            result = await self.test_single_report_speed(report_type)
            results[report_type] = result
            
            # Small delay to avoid overwhelming the server
            await asyncio.sleep(0.1)
        
        return results

    async def test_sustained_load(self, duration_seconds: int = 60, reports_per_second: int = 2) -> Dict[str, Any]:
        """Test sustained report generation load"""
        logger.info(f"⏱️ Testing sustained load: {reports_per_second} reports/sec for {duration_seconds} seconds...")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        total_reports = 0
        successful_reports = 0
        failed_reports = 0
        generation_times = []
        
        while time.time() < end_time:
            batch_start = time.time()
            
            # Generate reports for this second
            tasks = []
            for _ in range(reports_per_second):
                task = self.test_single_report_speed("performance")
                tasks.append(task)
            
            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process batch results
            for result in batch_results:
                total_reports += 1
                if isinstance(result, dict) and result.get("success"):
                    successful_reports += 1
                    generation_times.append(result["generation_time"])
                else:
                    failed_reports += 1
            
            # Wait for next second
            batch_time = time.time() - batch_start
            if batch_time < 1.0:
                await asyncio.sleep(1.0 - batch_time)
        
        total_duration = time.time() - start_time
        
        return {
            "duration_seconds": total_duration,
            "target_rps": reports_per_second,
            "total_reports": total_reports,
            "successful_reports": successful_reports,
            "failed_reports": failed_reports,
            "actual_rps": total_reports / total_duration,
            "success_rate": (successful_reports / total_reports) * 100 if total_reports > 0 else 0,
            "avg_generation_time": statistics.mean(generation_times) if generation_times else 0,
            "median_generation_time": statistics.median(generation_times) if generation_times else 0
        }

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            async with self.session.get(f"{self.base_url}/api/reports/metrics") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive performance test suite"""
        logger.info("🚀 Starting comprehensive report generation performance test...")
        
        test_results = {
            "test_start_time": datetime.now().isoformat(),
            "system_metrics": await self.get_system_metrics(),
            "tests": {}
        }
        
        # Test 1: Single report speed
        logger.info("\n" + "="*50)
        logger.info("TEST 1: Single Report Generation Speed")
        logger.info("="*50)
        test_results["tests"]["single_report"] = await self.test_single_report_speed()
        
        # Test 2: Different report types
        logger.info("\n" + "="*50)
        logger.info("TEST 2: Different Report Types")
        logger.info("="*50)
        test_results["tests"]["report_types"] = await self.test_different_report_types()
        
        # Test 3: Concurrent reports (10)
        logger.info("\n" + "="*50)
        logger.info("TEST 3: 10 Concurrent Reports")
        logger.info("="*50)
        test_results["tests"]["concurrent_10"] = await self.test_concurrent_reports(10)
        
        # Test 4: Concurrent reports (25)
        logger.info("\n" + "="*50)
        logger.info("TEST 4: 25 Concurrent Reports")
        logger.info("="*50)
        test_results["tests"]["concurrent_25"] = await self.test_concurrent_reports(25)
        
        # Test 5: Concurrent reports (50)
        logger.info("\n" + "="*50)
        logger.info("TEST 5: 50 Concurrent Reports")
        logger.info("="*50)
        test_results["tests"]["concurrent_50"] = await self.test_concurrent_reports(50)
        
        # Test 6: Bulk reports
        logger.info("\n" + "="*50)
        logger.info("TEST 6: Bulk Report Generation")
        logger.info("="*50)
        test_results["tests"]["bulk_reports"] = await self.test_bulk_reports(5)
        
        # Test 7: Sustained load (2 reports/sec for 30 seconds)
        logger.info("\n" + "="*50)
        logger.info("TEST 7: Sustained Load (2 reports/sec for 30s)")
        logger.info("="*50)
        test_results["tests"]["sustained_load"] = await self.test_sustained_load(30, 2)
        
        test_results["test_end_time"] = datetime.now().isoformat()
        
        return test_results

def print_test_summary(results: Dict[str, Any]):
    """Print comprehensive test summary"""
    print("\n" + "="*80)
    print("🚀 REPORT GENERATION PERFORMANCE TEST SUMMARY")
    print("="*80)
    
    # System metrics
    if "system_metrics" in results:
        metrics = results["system_metrics"]
        print(f"\n📊 SYSTEM CAPABILITIES:")
        print(f"   • Max Reports/Minute: {metrics.get('reports_per_minute', 'N/A')}")
        print(f"   • Average Generation Time: {metrics.get('avg_generation_time', 'N/A')}s")
        print(f"   • Cache Hit Rate: {metrics.get('cache_hit_rate', 'N/A')}%")
        print(f"   • Active Workers: {metrics.get('active_workers', 'N/A')}")
    
    tests = results.get("tests", {})
    
    # Single report performance
    if "single_report" in tests:
        single = tests["single_report"]
        if single.get("success"):
            print(f"\n⚡ SINGLE REPORT PERFORMANCE:")
            print(f"   • Generation Time: {single['generation_time']:.3f}s")
            print(f"   • Server Processing: {single['server_time']:.3f}s")
            print(f"   • Cache Hit: {'Yes' if single['cache_hit'] else 'No'}")
    
    # Concurrent performance
    for test_name in ["concurrent_10", "concurrent_25", "concurrent_50"]:
        if test_name in tests:
            concurrent = tests[test_name]
            count = concurrent.get("concurrent_count", 0)
            print(f"\n🔥 {count} CONCURRENT REPORTS:")
            print(f"   • Success Rate: {concurrent.get('success_rate', 0):.1f}%")
            print(f"   • Reports/Second: {concurrent.get('reports_per_second', 0):.1f}")
            print(f"   • Avg Generation Time: {concurrent.get('avg_generation_time', 0):.3f}s")
            print(f"   • Cache Hit Rate: {concurrent.get('cache_hit_rate', 0):.1f}%")
    
    # Sustained load
    if "sustained_load" in tests:
        sustained = tests["sustained_load"]
        print(f"\n⏱️ SUSTAINED LOAD TEST:")
        print(f"   • Duration: {sustained.get('duration_seconds', 0):.1f}s")
        print(f"   • Total Reports: {sustained.get('total_reports', 0)}")
        print(f"   • Success Rate: {sustained.get('success_rate', 0):.1f}%")
        print(f"   • Actual RPS: {sustained.get('actual_rps', 0):.1f}")
        print(f"   • Avg Generation Time: {sustained.get('avg_generation_time', 0):.3f}s")
    
    # Calculate estimated reports per minute
    best_rps = 0
    for test_name in ["concurrent_10", "concurrent_25", "concurrent_50"]:
        if test_name in tests:
            rps = tests[test_name].get("reports_per_second", 0)
            if rps > best_rps:
                best_rps = rps
    
    estimated_rpm = best_rps * 60
    print(f"\n🎯 PERFORMANCE SUMMARY:")
    print(f"   • Peak Reports/Second: {best_rps:.1f}")
    print(f"   • Estimated Reports/Minute: {estimated_rpm:.0f}")
    print(f"   • System Status: {'🟢 EXCELLENT' if estimated_rpm > 100 else '🟡 GOOD' if estimated_rpm > 50 else '🔴 NEEDS OPTIMIZATION'}")
    
    print("\n" + "="*80)

async def main():
    """Main test execution function"""
    parser = argparse.ArgumentParser(description="Report Generation Performance Test")
    parser.add_argument("--url", default="http://127.0.0.1:10000", help="Base URL for API")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--concurrent", type=int, default=25, help="Number of concurrent reports to test")
    parser.add_argument("--duration", type=int, default=30, help="Sustained load test duration")
    
    args = parser.parse_args()
    
    async with ReportPerformanceTester(args.url) as tester:
        if args.quick:
            # Quick test
            logger.info("🚀 Running quick performance test...")
            result = await tester.test_concurrent_reports(args.concurrent)
            print(f"\n⚡ QUICK TEST RESULTS:")
            print(f"   • {args.concurrent} Concurrent Reports")
            print(f"   • Success Rate: {result.get('success_rate', 0):.1f}%")
            print(f"   • Reports/Second: {result.get('reports_per_second', 0):.1f}")
            print(f"   • Estimated Reports/Minute: {result.get('reports_per_second', 0) * 60:.0f}")
        else:
            # Comprehensive test
            results = await tester.run_comprehensive_test()
            
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_performance_test_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print_test_summary(results)
            print(f"\n📄 Detailed results saved to: {filename}")

if __name__ == "__main__":
    asyncio.run(main())