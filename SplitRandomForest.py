from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('stock_data.csv')

# Chuẩn bị dữ liệu
df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y').timestamp())
X = df[['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume']]
y = df['Close']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Khởi tạo mô hình Random Forest
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# Huấn luyện mô hình trên tập huấn luyện
model.fit(X_train, y_train)

# Đưa ra dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
# Khởi tạo mô hình Random Forest
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# Đánh giá hiệu suất mô hình bằng cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Ước lượng mô hình
print('R-square:', r2_score(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
print('Cross-validation MSE:', np.mean(-scores), np.std(-scores))

