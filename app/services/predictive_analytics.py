
import logging
from datetime import datetime, timedelta
from services.mongodb_optimized_service import get_optimized_mongodb_service

logger = logging.getLogger(__name__)

class PredictiveAnalyticsService:
    """
    Advanced AI Service for generating issue forecasts and trend analysis.
    Uses generic regression models (Polynomial Regression) to predict future issue volume based on historical data.
    
    NOTE: pandas/numpy/sklearn are lazy-imported inside methods to save ~300MB RAM at startup.
    This is critical for Render's 512MB free-tier memory limit.
    """

    @staticmethod
    async def get_issue_forecast(days_history=30, days_forecast=7):
        """
        Analyzes the last `days_history` of issue data to predict the next `days_forecast` volume.
        Returns formatted data for frontend charts.
        """
        try:
            mongo = await get_optimized_mongodb_service()
            if not mongo:
                return PredictiveAnalyticsService._get_fallback_data()
            
            # 1. Fetch historical issue timestamps
            collection = await mongo.get_collection("issues")
            start_date = datetime.utcnow() - timedelta(days=days_history)
            
            issues = await collection.find(
                {"timestamp": {"$gte": start_date}},
                {"timestamp": 1, "_id": 0}
            ).to_list(None)

            if not issues or len(issues) < 5:
                # Not enough data for meaningful regression, return smart fallback/mock
                # But labeled as "Calibration Mode"
                logger.warning("Not enough data for prediction model. Using calibration fallback.")
                return PredictiveAnalyticsService._get_fallback_data()

            # 🚀 LAZY IMPORT: Only load heavy ML libs when actually needed (~300MB saved at startup)
            import pandas as pd
            import numpy as np
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import PolynomialFeatures

            # 2. Process Data into Time Series (Daily counts)
            df = pd.DataFrame(issues)
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_counts = df.groupby('date').size().reset_index(name='count')
            
            # Fill missing days with 0 to ensure continuity
            all_days = pd.date_range(start=start_date.date(), end=datetime.utcnow().date())
            daily_counts = daily_counts.set_index('date').reindex(all_days.date, fill_value=0).reset_index()
            daily_counts.columns = ['date', 'count']

            # 3. Feature Engineering for ML
            # X = Days since start, y = Issue Count
            daily_counts['day_index'] = (pd.to_datetime(daily_counts['date']) - pd.to_datetime(start_date.date())).dt.days
            X = daily_counts[['day_index']].values
            y = daily_counts['count'].values

            # 4. Train Model (Polynomial Regression for non-linear trends)
            # Degree 2 captures curves (e.g. increasing trend) better than simple lines
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, y)

            # 5. Generate Predictions (Past + Future)
            last_day_index = X[-1][0]
            future_days = np.arange(last_day_index + 1, last_day_index + 1 + days_forecast).reshape(-1, 1)
            
            # Predict
            X_future_poly = poly.transform(future_days)
            future_predictions = model.predict(X_future_poly)
            
            # Ensure no negative predictions
            future_predictions = np.maximum(future_predictions, 0)

            # 6. Format Response
            chart_data = []
            
            # Add recent history (last 7 days actuals) to chart
            recent_actuals = daily_counts.tail(7)
            for _, row in recent_actuals.iterrows():
                chart_data.append({
                    "name": row['date'].strftime('%a'), # Mon, Tue
                    "full_date": row['date'].strftime('%Y-%m-%d'),
                    "actual": int(row['count']),
                    "predicted": int(model.predict(poly.transform([[row['day_index']]]))[0]) # Smooth curve overlay
                })
                
            # Add future predictions
            next_date = datetime.utcnow().date()
            for i, pred_val in enumerate(future_predictions):
                next_day = next_date + timedelta(days=i+1)
                chart_data.append({
                    "name": next_day.strftime('%a'),
                    "full_date": next_day.strftime('%Y-%m-%d'),
                    "actual": None, # Future has no actuals yet
                    "predicted": int(round(pred_val)),
                    "is_future": True
                })

            # Calculate Trend Percentage
            avg_past = np.mean(y[-7:]) if len(y) >= 7 else np.mean(y)
            avg_future = np.mean(future_predictions)
            trend_pct = ((avg_future - avg_past) / (avg_past + 0.1)) * 100 # +0.1 to avoid div by zero

            return {
                "chart_data": chart_data,
                "trend_percentage": round(trend_pct, 1),
                "trend_direction": "up" if trend_pct > 0 else "down",
                "model_status": "active"
            }

        except Exception as e:
            logger.error(f"Prediction model error: {e}", exc_info=True)
            return PredictiveAnalyticsService._get_fallback_data()

    @staticmethod
    def _get_fallback_data():
        """Returns stylized fallback data if DB is empty or error occurs"""
        return {
             "chart_data": [
                { "name": 'Mon', "actual": 12, "predicted": 12 },
                { "name": 'Tue', "actual": 19, "predicted": 16 },
                { "name": 'Wed', "actual": 15, "predicted": 18 },
                { "name": 'Thu', "actual": 22, "predicted": 21 },
                { "name": 'Fri', "actual": 30, "predicted": 25 },
                { "name": 'Sat', "actual": 18, "predicted": 20 },
                { "name": 'Sun', "actual": 0, "predicted": 22, "is_future": True },
            ],
            "trend_percentage": 10.5,
            "trend_direction": "up",
            "model_status": "calibration"
        }

