import pandas as pd
import sqlite3
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
conn = sqlite3.connect("data/olist.db")
df = pd.read_sql_query("SELECT * FROM master_orders", conn)
conn.close()
print(f"Loaded {len(df)} rows")

# ── FEATURE ENGINEERING ─────────────────────────────────────────────
print("\n── Engineering Features ──")
df['freight_ratio'] = df['freight_value'] / (df['price'] + 1)
df['is_holiday_season'] = df['order_month'].apply(
    lambda x: 1 if x in [11, 12] else 0
)
df['seller_customer_same_state'] = (
    df['seller_state'] == df['customer_state']
).astype(int)
print("Features added: freight_ratio, is_holiday_season, seller_customer_same_state")

# ── MODEL 1: DELIVERY DELAY PREDICTOR ──────────────────────────────
print("\n── Training Delivery Delay Predictor ──")

delay_features = [
    'customer_state', 'seller_state', 'category',
    'price', 'freight_value', 'payment_value',
    'order_month', 'order_year',
    'freight_ratio', 'is_holiday_season', 'seller_customer_same_state'
]
delay_target = 'delay_flag'

df_delay = df[delay_features + [delay_target]].dropna().copy()
print(f"Training rows: {len(df_delay)} | Delay rate: {df_delay[delay_target].mean():.1%}")

le_customer_state = LabelEncoder()
le_seller_state = LabelEncoder()
le_category = LabelEncoder()

df_delay['customer_state'] = le_customer_state.fit_transform(df_delay['customer_state'])
df_delay['seller_state'] = le_seller_state.fit_transform(df_delay['seller_state'])
df_delay['category'] = le_category.fit_transform(df_delay['category'].astype(str))

X_delay = df_delay[delay_features]
y_delay = df_delay[delay_target]

X_train, X_test, y_train, y_test = train_test_split(
    X_delay, y_delay, test_size=0.2, random_state=42
)

try:
    from xgboost import XGBClassifier
    print("Training XGBoost...")
    delay_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=int((y_train==0).sum()/(y_train==1).sum()),
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    delay_model.fit(X_train, y_train)
    model_used = "XGBoost"
except ImportError:
    print("Using RandomForest...")
    delay_model = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=42, n_jobs=-1, class_weight='balanced'
    )
    delay_model.fit(X_train, y_train)
    model_used = "RandomForest"

y_pred = delay_model.predict(X_test)
print(f"Model: {model_used} | Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, target_names=['On Time', 'Delayed']))

# ── MODEL 2: REVIEW SCORE PREDICTOR ────────────────────────────────
print("\n── Training Review Score Predictor ──")

df['review_bucket'] = pd.cut(
    df['review_score'], bins=[0, 2, 3, 5],
    labels=['Low (1-2)', 'Medium (3)', 'High (4-5)']
)

review_features = [
    'delivery_days', 'delay_flag', 'freight_value',
    'price', 'payment_value', 'order_month',
    'freight_ratio', 'is_holiday_season'
]

df_review = df[review_features + ['review_bucket']].dropna().copy()
print(f"Training rows: {len(df_review)}")

X_review = df_review[review_features]
y_review = df_review['review_bucket']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_review, y_review, test_size=0.2, random_state=42
)

print("Training RandomForest...")
review_model = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    random_state=42, n_jobs=-1, class_weight='balanced'
)
review_model.fit(X_train_r, y_train_r)
y_pred_r = review_model.predict(X_test_r)
print(f"Accuracy: {accuracy_score(y_test_r, y_pred_r):.3f}")
print(classification_report(y_test_r, y_pred_r))

print("Top Features:")
importances_r = pd.Series(
    review_model.feature_importances_, index=review_features
).sort_values(ascending=False)
print(importances_r.head())

# ── MODEL 3: CUSTOMER SEGMENTATION (RFM + KMeans) ──────────────────
print("\n── Training Customer Segmentation ──")

# Calculate RFM metrics per customer
df['order_date'] = pd.to_datetime(df['order_date'])
snapshot_date = df['order_date'].max() + pd.Timedelta(days=1)

rfm = df.groupby('order_id').agg(
    order_date=('order_date', 'max'),
    monetary=('payment_value', 'sum')
).reset_index()

# Get customer level
rfm_customer = df.groupby('order_id').agg(
    order_date=('order_date', 'max'),
    monetary=('payment_value', 'sum')
).reset_index()

# Since we have order_id not customer_id directly, use order level RFM
rfm = df.groupby('order_id').agg(
    recency_date=('order_date', 'max'),
    monetary=('payment_value', 'sum'),
    frequency=('order_id', 'count')
).reset_index()

rfm['recency'] = (snapshot_date - rfm['recency_date']).dt.days
rfm = rfm[['order_id', 'recency', 'frequency', 'monetary']].dropna()

print(f"RFM table: {len(rfm)} orders")
print(f"Recency range: {rfm['recency'].min()} - {rfm['recency'].max()} days")
print(f"Monetary range: R${rfm['monetary'].min():.0f} - R${rfm['monetary'].max():.0f}")

# Scale features for KMeans
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])

# Find optimal k using inertia
print("Finding optimal number of clusters...")
inertias = []
k_range = range(2, 8)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertias.append(km.inertia_)

# Use k=4 — standard RFM segmentation
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['segment'] = kmeans.fit_predict(rfm_scaled)

# Label segments based on RFM characteristics
segment_summary = rfm.groupby('segment').agg(
    count=('order_id', 'count'),
    avg_recency=('recency', 'mean'),
    avg_frequency=('frequency', 'mean'),
    avg_monetary=('monetary', 'mean')
).round(2)

print("\nSegment Summary:")
print(segment_summary)

# Auto-label segments
def label_segment(row):
    if row['avg_monetary'] > 5000:
        return 'VIP / Whale'
    elif row['avg_recency'] < 200 and row['avg_monetary'] > 150:
        return 'Champions'
    elif row['avg_frequency'] > 3 and row['avg_monetary'] > 500:
        return 'Loyal Customers'
    elif row['avg_recency'] > 350:
        return 'At Risk / Hibernating'
    else:
        return 'New / Occasional'

segment_summary['label'] = segment_summary.apply(label_segment, axis=1)
print("\nSegment Labels:")
print(segment_summary[['count', 'label']])


# ── SAVE ALL MODELS ─────────────────────────────────────────────────
print("\n── Saving All Models ──")
os.makedirs("models", exist_ok=True)

joblib.dump(delay_model, "models/delay_model.pkl")
joblib.dump(review_model, "models/review_model.pkl")
joblib.dump(le_customer_state, "models/le_customer_state.pkl")
joblib.dump(le_seller_state, "models/le_seller_state.pkl")
joblib.dump(le_category, "models/le_category.pkl")
joblib.dump(delay_features, "models/delay_features.pkl")
joblib.dump(review_features, "models/review_features.pkl")
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(scaler,"models/rfm_scaler.pkl")
joblib.dump(rfm, "models/rfm_data.pkl")
joblib.dump(segment_summary, "models/segment_summary.pkl")

print("All models saved to models/ folder")
print("\nDone!")