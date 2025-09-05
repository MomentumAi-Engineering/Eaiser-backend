# 🚀 Celery Configuration for SnapFix Background Tasks
# Handles async processing for 1 lakh concurrent users

from celery import Celery
from celery.signals import worker_ready, worker_shutdown
import os
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Celery Configuration
class CeleryConfig:
    """Celery configuration optimized for high throughput"""
    
    # Broker settings (RabbitMQ)
    broker_url = os.getenv('RABBITMQ_URL', 'amqp://snapfix:snapfix123@localhost:5672//')
    result_backend = os.getenv('REDIS_URL', 'redis://localhost:6379/1')
    
    # Task settings for high performance
    task_serializer = 'json'
    accept_content = ['json']
    result_serializer = 'json'
    timezone = 'UTC'
    enable_utc = True
    
    # Worker settings for 1 lakh users
    worker_prefetch_multiplier = 4  # Prefetch 4 tasks per worker
    task_acks_late = True          # Acknowledge after task completion
    worker_disable_rate_limits = False
    
    # Task routing for different priorities
    task_routes = {
        'celery_app.process_issue_report': {'queue': 'high_priority'},
        'celery_app.send_notification_email': {'queue': 'medium_priority'},
        'celery_app.process_image_upload': {'queue': 'low_priority'},
        'celery_app.generate_analytics': {'queue': 'low_priority'},
    }
    
    # Queue configuration
    task_default_queue = 'default'
    task_queues = {
        'high_priority': {
            'exchange': 'high_priority',
            'exchange_type': 'direct',
            'routing_key': 'high_priority',
        },
        'medium_priority': {
            'exchange': 'medium_priority',
            'exchange_type': 'direct',
            'routing_key': 'medium_priority',
        },
        'low_priority': {
            'exchange': 'low_priority',
            'exchange_type': 'direct',
            'routing_key': 'low_priority',
        }
    }
    
    # Task execution settings
    task_soft_time_limit = 300     # 5 minutes soft limit
    task_time_limit = 600          # 10 minutes hard limit
    task_max_retries = 3           # Retry failed tasks 3 times
    task_default_retry_delay = 60  # Wait 60 seconds before retry
    
    # Result backend settings
    result_expires = 3600          # Results expire after 1 hour
    result_backend_transport_options = {
        'master_name': 'mymaster',
        'visibility_timeout': 3600,
    }
    
    # Monitoring and logging
    worker_send_task_events = True
    task_send_sent_event = True
    
    # Performance optimizations
    broker_connection_retry_on_startup = True
    broker_connection_retry = True
    broker_connection_max_retries = 10

# Initialize Celery app
celery_app = Celery('snapfix')
celery_app.config_from_object(CeleryConfig)

# Import task modules
celery_app.autodiscover_tasks(['celery_app'])

@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Called when worker is ready to receive tasks"""
    logger.info(f"SnapFix Celery worker {sender} is ready for high-load processing!")

@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Called when worker is shutting down"""
    logger.info(f"SnapFix Celery worker {sender} is shutting down gracefully")

# Task definitions
@celery_app.task(bind=True, max_retries=3)
def process_issue_report(self, issue_data: Dict[str, Any]):
    """Process issue report in background - HIGH PRIORITY
    
    Args:
        issue_data: Dictionary containing issue information
        
    Returns:
        Dict with processing status
    """
    try:
        logger.info(f"Processing issue report: {issue_data.get('issue_id')}")
        
        # Import here to avoid circular imports
        from services.mongodb_service import save_issue
        from services.notification_service import send_issue_notification
        
        # Save issue to database
        result = save_issue(issue_data)
        
        # Send notification asynchronously
        if result.get('success'):
            send_notification_email.delay(
                issue_data.get('user_email'),
                issue_data.get('issue_id'),
                'issue_created'
            )
        
        logger.info(f"Issue {issue_data.get('issue_id')} processed successfully")
        return {
            'status': 'success',
            'issue_id': issue_data.get('issue_id'),
            'processed_at': str(self.request.id)
        }
        
    except Exception as exc:
        logger.error(f"Error processing issue {issue_data.get('issue_id')}: {str(exc)}")
        
        # Retry with exponential backoff
        countdown = 2 ** self.request.retries
        raise self.retry(exc=exc, countdown=countdown, max_retries=3)

@celery_app.task(bind=True, max_retries=3)
def send_notification_email(self, user_email: str, issue_id: str, notification_type: str):
    """Send notification email - MEDIUM PRIORITY
    
    Args:
        user_email: User's email address
        issue_id: Issue identifier
        notification_type: Type of notification
        
    Returns:
        Dict with email status
    """
    try:
        logger.info(f"Sending {notification_type} email to {user_email} for issue {issue_id}")
        
        # Import here to avoid circular imports
        from services.email_service import send_email
        
        # Email templates based on notification type
        templates = {
            'issue_created': {
                'subject': f'Issue #{issue_id} Created Successfully',
                'body': f'Your issue #{issue_id} has been created and is being processed.'
            },
            'issue_updated': {
                'subject': f'Issue #{issue_id} Status Updated',
                'body': f'Your issue #{issue_id} status has been updated.'
            },
            'issue_resolved': {
                'subject': f'Issue #{issue_id} Resolved',
                'body': f'Great news! Your issue #{issue_id} has been resolved.'
            }
        }
        
        template = templates.get(notification_type, templates['issue_created'])
        
        # Send email
        result = send_email(
            to_email=user_email,
            subject=template['subject'],
            body=template['body']
        )
        
        logger.info(f"Email sent successfully to {user_email}")
        return {
            'status': 'success',
            'email': user_email,
            'issue_id': issue_id,
            'notification_type': notification_type
        }
        
    except Exception as exc:
        logger.error(f"Error sending email to {user_email}: {str(exc)}")
        
        # Retry with exponential backoff
        countdown = 2 ** self.request.retries
        raise self.retry(exc=exc, countdown=countdown, max_retries=3)

@celery_app.task(bind=True, max_retries=2)
def process_image_upload(self, image_data: Dict[str, Any]):
    """Process and optimize uploaded images - LOW PRIORITY
    
    Args:
        image_data: Dictionary containing image information
        
    Returns:
        Dict with processing status
    """
    try:
        logger.info(f"Processing image upload: {image_data.get('filename')}")
        
        # Import here to avoid circular imports
        from services.image_service import optimize_image, generate_thumbnails
        
        # Optimize image
        optimized_path = optimize_image(image_data['file_path'])
        
        # Generate thumbnails
        thumbnails = generate_thumbnails(optimized_path)
        
        logger.info(f"Image {image_data.get('filename')} processed successfully")
        return {
            'status': 'success',
            'original_path': image_data['file_path'],
            'optimized_path': optimized_path,
            'thumbnails': thumbnails
        }
        
    except Exception as exc:
        logger.error(f"Error processing image {image_data.get('filename')}: {str(exc)}")
        
        # Retry with exponential backoff
        countdown = 2 ** self.request.retries
        raise self.retry(exc=exc, countdown=countdown, max_retries=2)

@celery_app.task(bind=True)
def generate_analytics(self, analytics_type: str, date_range: Dict[str, str]):
    """Generate analytics reports - LOW PRIORITY
    
    Args:
        analytics_type: Type of analytics to generate
        date_range: Date range for analytics
        
    Returns:
        Dict with analytics data
    """
    try:
        logger.info(f"Generating {analytics_type} analytics for {date_range}")
        
        # Import here to avoid circular imports
        from services.analytics_service import generate_report
        
        # Generate analytics report
        report = generate_report(analytics_type, date_range)
        
        logger.info(f"Analytics {analytics_type} generated successfully")
        return {
            'status': 'success',
            'analytics_type': analytics_type,
            'date_range': date_range,
            'report': report
        }
        
    except Exception as exc:
        logger.error(f"Error generating analytics {analytics_type}: {str(exc)}")
        return {
            'status': 'error',
            'error': str(exc)
        }

# Health check task
@celery_app.task
def health_check():
    """Health check task for monitoring"""
    return {
        'status': 'healthy',
        'timestamp': str(celery_app.now()),
        'worker_id': str(celery_app.control.inspect().active())
    }

if __name__ == '__main__':
    celery_app.start()