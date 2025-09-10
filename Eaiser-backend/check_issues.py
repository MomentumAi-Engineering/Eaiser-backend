import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_issues():
    try:
        # Connect to MongoDB
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
        db_name = os.getenv('MONGODB_NAME', 'eaiser_db')
        
        client = MongoClient(mongo_uri)
        db = client[db_name]
        
        # Get issues
        issues = list(db.issues.find({}, {'_id': 1}).limit(5))
        print('Available issue IDs:')
        for issue in issues:
            print(str(issue['_id']))
        
        if not issues:
            print('No issues found in database')
            
        client.close()
            
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    check_issues()