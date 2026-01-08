import uvicorn
import os
import multiprocessing
import sys

# Add app directory to Python path
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, app_dir)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    # Calculate workers: Number of CPUs
    # Cap at 4 for stability on Windows/MongoDB Atlas to prevent connection saturation
    try:
        workers = min(4, multiprocessing.cpu_count()) 
    except:
        workers = 4

    print(f"\n🚀 STARTING HIGH-PERFORMANCE CLUSTER")
    print(f"====================================")
    print(f"⚡ WORKERS (SERVERS): {workers}")
    print(f"⚡ PORT:              {port}")
    print(f"⚡ OPTIMIZATIONS:     ENABLED (HTTPTools, Auto Loop)")
    print(f"====================================\n")

    # Run Uvicorn with workers
    # 'workers' > 1 enables the process manager which acts as a load balancer
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        workers=workers,
        log_level="info",
        loop="auto",
        http="auto"
    )
