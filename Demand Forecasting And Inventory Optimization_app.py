
# Scope for project
Demand Forecasting involves predicting the quantity and pattern of customer orders, which is crucial for businesses to efficiently allocate resources, manage inventory, and plan production. 
Accurate demand forecasting enables companies to meet customer needs, avoid overstocking or understocking, and optimize their supply chain operations.
Inventory Optimization aims to strike a balance between having sufficient stock to meet demand without carrying excess inventory that ties up capital and storage space. Effective inventory 
optimization helps businesses reduce carrying costs, improve cash flow, and enhance customer satisfaction.
These concepts are especially relevant in retail, manufacturing, and distribution, where managing supply and demand dynamics is essential for profitability and customer satisfaction.

# Techniques/Tools For demand forecasting, we can make use of suitable forecasting models such as exponential smoothing, SARIMA, and ARIMA. After that, we can use the demand projections to 
optimize inventory levels through the application of techniques such as safety stock, reorder points, and economic order quantity (EOQ) computations.

# importing libraries and loading of dataset
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

data = pd.read_csv("demand_inventory.csv")
data.head()

data = data.drop(columns=['Unnamed: 0'])

fig_demand = px.line(data, x='Date',
                     y='Demand',
                     title='Demand Over Time')
fig_demand.show()

fig_inventory = px.line(data, x='Date',
                        y='Inventory',
                        title='Inventory Over Time')
fig_inventory.show()

#forecast the demand using SARIMA. Let’s first calculate the value of p and q using ACF and PACF plots:

from datetime import datetime

date_string = "2023/06/01"
specified_format = "%Y/%m/%d"

# Convert the date string to a datetime object
date_object = datetime.strptime(date_string, specified_format)

# Now you can use the date_object as needed
print(date_object)

data['Date'] = pd.to_datetime(data['Date'].str.replace('-', '/'), format='%Y/%m/%d')

time_series = data.set_index('Date')['Demand']

differenced_series = time_series.diff().dropna()

# Plot ACF and PACF of differenced time series
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(differenced_series, ax=axes[0])
plot_pacf(differenced_series, ax=axes[1])
plt.show()

# let’s train the model and forecast demand for the next ten days:
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 2) #2 because the data contains a time period of 2 months only
model = SARIMAX(time_series, order=order, seasonal_order=seasonal_order)
model_fit = model.fit(disp=False)

future_steps = 10
predictions = model_fit.predict(len(time_series), len(time_series) + future_steps - 1)
predictions = predictions.astype(int)
print(predictions)
# out:
2023-08-02    117
2023-08-03    116
2023-08-04    130
2023-08-05    114
2023-08-06    128
2023-08-07    115
2023-08-08    129
2023-08-09    115
2023-08-10    129
2023-08-11    115
Freq: D, Name: predicted_mean, dtype: int32

#  optimize inventory according to the forecasted demand for the next ten days:

# Create date indices for the future predictions
future_dates = pd.date_range(start=time_series.index[-1] + pd.DateOffset(days=1), periods=future_steps, freq='D')

# Create a pandas Series with the predicted values and date indices
forecasted_demand = pd.Series(predictions, index=future_dates)

# Initial inventory level
initial_inventory = 5500

# Lead time (number of days it takes to replenish inventory) 
lead_time = 1 # it's different for every business, 1 is an example

# Service level (probability of not stocking out)
service_level = 0.95 # it's different for every business, 0.95 is an example

# Calculate the optimal order quantity using the Newsvendor formula
z = np.abs(np.percentile(forecasted_demand, 100 * (1 - service_level)))
order_quantity = np.ceil(forecasted_demand.mean() + z).astype(int)

# Calculate the reorder point
reorder_point = forecasted_demand.mean() * lead_time + z

# Calculate the optimal safety stock
safety_stock = reorder_point - forecasted_demand.mean() * lead_time

# Calculate the total cost (holding cost + stockout cost)
holding_cost = 0.1  # it's different for every business, 0.1 is an example
stockout_cost = 10  # # it's different for every business, 10 is an example
total_holding_cost = holding_cost * (initial_inventory + 0.5 * order_quantity)
total_stockout_cost = stockout_cost * np.maximum(0, forecasted_demand.mean() * lead_time - initial_inventory)

# Calculate the total cost
total_cost = total_holding_cost + total_stockout_cost

print("Optimal Order Quantity:", order_quantity)
print("Reorder Point:", reorder_point)
print("Safety Stock:", safety_stock)
print("Total Cost:", total_cost)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Function to forecast demand and optimize inventory
def forecast_demand_and_optimize_inventory(data):
    # Your demand forecasting and inventory optimization code here
    # This function should return the optimal order quantity, reorder point, safety stock, and total cost
    
    return optimal_order_quantity, reorder_point, safety_stock, total_cost

# Load the dataset
data = pd.read_csv("demand_inventory.csv")

# Streamlit UI
st.title('Demand Forecasting and Inventory Optimization')

# Display the dataset
st.subheader('Dataset')
st.write(data)

# Perform demand forecasting and inventory optimization
optimal_order_quantity, reorder_point, safety_stock, total_cost = forecast_demand_and_optimize_inventory(data)

# Display the results
st.subheader('Optimal Order Quantity:')
st.write(optimal_order_quantity)

st.subheader('Reorder Point:')
st.write(reorder_point)

st.subheader('Safety Stock:')
st.write(safety_stock)

st.subheader('Total Cost:')
st.write(total_cost)

if __name__ == '__main__':
    main()


# out:
#Optimal Order Quantity: 236
#Reorder Point: 235.25
#Safety Stock: 114.45
#Total Cost: 561.8000000000001

#Optimal Order Quantity: 236 – The amount of a product that should be ordered from suppliers when the inventory level hits a particular threshold is known as the optimal order quantity. 
#An ideal order quantity of 236 units has been determined in this instance.

#Reorder Point: 235.25 – The inventory level at which a fresh order needs to be made in order to refill stock before it runs out is known as the reorder point. A reorder point of 235.25 units 
#has been determined in this instance, meaning that an order for additional stock should be placed when the inventory reaches or drops below this threshold.

#Safety Stock: 114.45 – The extra inventory held on hand to cover supply and demand fluctuations is known as safety stock. It serves as a safeguard against unforeseen changes in lead time or 
#demand. A safety stock of 114.45 units has been computed in this instance to assist make sure that there is adequate inventory to handle any variations in lead time or demand.

#Total Cost: 561.80 –The total cost is the sum of all the expenses related to inventory control. This order's total cost has been estimated at 561.80 units based on safety stock, reorder point, 
#order quantity, and related expenses.

#Demand Forecasting involves predicting the quantity and pattern of customer orders, which is crucial for businesses to efficiently allocate resources, manage inventory, and plan production. 
#Inventory Optimization aims to strike a balance between having sufficient stock to meet demand without carrying excess inventory that ties up capital and storage space. 
#In [ ]:


​
