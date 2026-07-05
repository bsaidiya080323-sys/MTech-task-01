"""
Charity Fundraising Campaign Optimizer - ML Models
Implements predictive models using scikit-learn for campaign optimization.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.cluster import KMeans
import joblib
import warnings
warnings.filterwarnings('ignore')

class CampaignMLModels:
    """Machine Learning models for charity campaign optimization."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.is_trained = False

    def _encode_categorical(self, df, columns, fit=True):
        """Encode categorical variables."""
        df_encoded = df.copy()
        for col in columns:
            if col not in df_encoded.columns:
                continue
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    df_encoded[col] = df_encoded[col].astype(str).apply(
                        lambda x: x if x in self.label_encoders[col].classes_ else self.label_encoders[col].classes_[0]
                    )
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        return df_encoded

    def prepare_features(self, df, target_col=None, fit=True):
        """Prepare features for ML models."""
        feature_cols = ['budget', 'duration_days', 'outreach_count', 'month', 'engagement_rate']
        categorical_cols = ['campaign_type', 'channel', 'target_audience', 'cause']

        available_cat = [c for c in categorical_cols if c in df.columns]
        available_num = [c for c in feature_cols if c in df.columns]

        df_processed = self._encode_categorical(df, available_cat, fit=fit)

        X = df_processed[available_num + available_cat]

        if target_col and target_col in df.columns:
            y = df_processed[target_col]
            return X, y
        return X

    def train_amount_predictor(self, df):
        """Train model to predict amount raised."""
        print("Training Amount Raised Predictor...")

        X, y = self.prepare_features(df, 'amount_raised', fit=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Gradient Boosting Regressor
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"  RMSE: ${np.sqrt(mse):,.2f}")
        print(f"  R² Score: {r2:.4f}")

        self.models['amount_predictor'] = model
        self.is_trained = True

        return {'rmse': np.sqrt(mse), 'r2': r2, 'model': model}

    def train_roi_predictor(self, df):
        """Train model to predict ROI."""
        print("Training ROI Predictor...")

        X, y = self.prepare_features(df, 'roi', fit=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"  RMSE: {np.sqrt(mse):.4f}")
        print(f"  R² Score: {r2:.4f}")

        self.models['roi_predictor'] = model
        self.is_trained = True

        # Feature importance
        feature_names = X.columns.tolist()
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return {'rmse': np.sqrt(mse), 'r2': r2, 'feature_importance': importance_df}

    def train_success_classifier(self, df):
        """Train model to classify campaign success."""
        print("Training Success Classifier...")

        X, y = self.prepare_features(df, 'success_label', fit=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"  Accuracy: {accuracy:.4f}")

        self.models['success_classifier'] = model
        self.is_trained = True

        return {'accuracy': accuracy, 'report': report}

    def train_donor_segmentation(self, df):
        """Segment donors using K-Means clustering."""
        print("Training Donor Segmentation Model...")

        features = ['age', 'income', 'years_donating', 'total_donations', 
                   'engagement_score', 'recency_months', 'frequency_per_year', 'avg_monetary']

        available = [c for c in features if c in df.columns]
        X = df[available].copy()
        X = X.fillna(X.mean())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Find optimal clusters
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        df['segment'] = clusters

        # Analyze segments
        segment_analysis = df.groupby('segment').agg({
            'age': 'mean',
            'income': 'mean',
            'total_donations': 'mean',
            'engagement_score': 'mean',
            'rfm_score': 'mean'
        }).round(2)

        segment_names = {
            0: 'Champions',
            1: 'Loyal Supporters', 
            2: 'Potential Donors',
            3: 'At-Risk Donors'
        }

        # Map based on RFM scores
        sorted_segments = segment_analysis.sort_values('rfm_score', ascending=False).index.tolist()
        name_map = {
            sorted_segments[0]: 'Champions',
            sorted_segments[1]: 'Loyal Supporters',
            sorted_segments[2]: 'Potential Donors',
            sorted_segments[3]: 'At-Risk Donors'
        }

        df['segment_name'] = df['segment'].map(name_map)

        print(f"  Segments created: {df['segment_name'].value_counts().to_dict()}")

        self.models['donor_segmentation'] = kmeans

        return {'segments': df, 'analysis': segment_analysis}

    def predict_campaign_performance(self, campaign_data):
        """Predict performance for a new campaign."""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction!")

        X = self.prepare_features(campaign_data, fit=False)
        X_scaled = self.scaler.transform(X)

        predictions = {}

        if 'amount_predictor' in self.models:
            predictions['amount_raised'] = self.models['amount_predictor'].predict(X_scaled)[0]

        if 'roi_predictor' in self.models:
            predictions['roi'] = self.models['roi_predictor'].predict(X_scaled)[0]

        if 'success_classifier' in self.models:
            predictions['success_label'] = self.models['success_classifier'].predict(X_scaled)[0]
            proba = self.models['success_classifier'].predict_proba(X_scaled)[0]
            classes = self.models['success_classifier'].classes_
            predictions['success_probabilities'] = dict(zip(classes, proba))

        return predictions

    def get_feature_importance(self):
        """Get feature importance from trained models."""
        if 'roi_predictor' not in self.models:
            return None

        feature_names = ['budget', 'duration_days', 'outreach_count', 'month', 'engagement_rate',
                        'campaign_type', 'channel', 'target_audience', 'cause']
        importances = self.models['roi_predictor'].feature_importances_

        return pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)


if __name__ == '__main__':
    from data_generator import CampaignDataGenerator

    gen = CampaignDataGenerator()
    campaigns = gen.generate_historical_campaigns(500)
    donors = gen.generate_donor_profiles(1000)

    ml = CampaignMLModels()
    ml.train_amount_predictor(campaigns)
    ml.train_roi_predictor(campaigns)
    ml.train_success_classifier(campaigns)
    ml.train_donor_segmentation(donors)
