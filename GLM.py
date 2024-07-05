import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    data['date'] = pd.to_datetime(data['datetime'])
    
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day  
    return data


def build_and_evaluate_model(data):
    X = data[['month', 'day', 'dew', 'humidity', 'uvindex', 'precip']]
    y = data['temp']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)
    
    model = sm.GLM(y_train, X_train_const, family=sm.families.Gaussian())
    results = model.fit()
    
    y_pred = results.predict(X_test_const)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2, results


def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
    plt.xlabel('Actual Temperature')
    plt.ylabel('Predicted Temperature')
    plt.title('Actual vs Predicted Temperature')
    plt.show()

    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, alpha=0.5, color='blue')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.show()


if __name__ == "__main__":
    data = load_and_preprocess_data('GLM_data.csv')
    mse, r2, results = build_and_evaluate_model(data)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    y_test, y_pred = results.model.endog, results.fittedvalues
    plot_results(y_test, y_pred)

