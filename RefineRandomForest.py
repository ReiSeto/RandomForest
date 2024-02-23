from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime
import pandas as pd
from joblib import dump

# Load data
df = pd.read_csv('stock_data.csv')

# Chuẩn bị dữ liệu
df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y').timestamp())
X = df[['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume']]
y = df['Close']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tạo một mô hình Random Forest
rf = RandomForestRegressor()

# Khai báo các giá trị của các tham số để duyệt qua
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': [1, 2, 4]
}

# Thực hiện tìm kiếm lưới
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)

# In ra giá trị tham số tốt nhất
print(grid_search.best_params_)

# Sử dụng giá trị tham số tốt nhất để huấn luyện mô hình
best_rf = RandomForestRegressor(n_estimators=grid_search.best_params_['n_estimators'], 
                                 max_depth=grid_search.best_params_['max_depth'], 
                                 max_features=grid_search.best_params_['max_features'], 
                                 min_samples_leaf=grid_search.best_params_['min_samples_leaf'])
best_rf.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = best_rf.predict(X_test)
print('R-square:', r2_score(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
