import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Load your dataset (replace '/path/to/your/dataset.csv' with the actual path)
df_countries = pd.read_csv('E:\ZK02_P2\API_EN.ATM.CO2E.LF.KT_DS2_en_csv_v2_6305458.csv', skiprows=4)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def co2_emissions_prediction(df_countries, countries):
    indicator_name = 'CO2 emissions from liquid fuel consumption (kt)'

    def co2_emissions_model(year, a, b, c):
        return a * np.exp(b * (year - 1990)) + c

    def plot_country_co2_emissions(country_data, country_name):
        # Extract years and CO2 emissions data for the specified country and indicator
        years = country_data.columns[4:]  # Assuming the years start from the 5th column
        co2_emissions = country_data.iloc[:, 4:].values.flatten()

        # Convert years to numeric values
        years_numeric = pd.to_numeric(years, errors='coerce')
        co2_emissions = pd.to_numeric(co2_emissions, errors='coerce')

        # Remove rows with NaN or inf values
        valid_data_mask = np.isfinite(years_numeric) & np.isfinite(co2_emissions)
        years_numeric = years_numeric[valid_data_mask]
        co2_emissions = co2_emissions[valid_data_mask]

        # Curve fitting with increased maxfev
        params, covariance = curve_fit(co2_emissions_model, years_numeric, co2_emissions, p0=[1, -0.1, 90], maxfev=10000)

        # Optimal parameters
        a_opt, b_opt, c_opt = params

        # Generate model predictions for the year 2040
        year_2040 = 2040
        co2_emissions_2040 = co2_emissions_model(year_2040, a_opt, b_opt, c_opt)

        # Plot the original data and the fitted curve
        plt.figure(figsize=(10, 6))
        plt.scatter(years_numeric, co2_emissions, label='Original Data', color='blue', alpha=0.7, edgecolors='black')
        plt.plot(years_numeric, co2_emissions_model(years_numeric, a_opt, b_opt, c_opt), label='Fitted Curve', color='red')

        # Highlight the prediction for 2040
        plt.scatter(year_2040, co2_emissions_2040, color='green', marker='*', label='Prediction for 2040', s=100, edgecolors='black')

        # Add labels and legend
        plt.title(f'CO2 Emissions Prediction for {country_name}')
        plt.xlabel('Year')
        plt.ylabel('CO2 Emissions (kt)')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

    for country_name in countries:
        country_data = df_countries[(df_countries['Country Name'] == country_name) & (df_countries['Indicator Name'] == indicator_name)]
        if not country_data.empty:
            plot_country_co2_emissions(country_data, country_name)

# Countries to plot
countries_to_plot = ['Argentina', 'Albania', 'China']
co2_emissions_prediction(df_countries, countries_to_plot)

# Extract data for the years 1999 and 2022
years = ['1999', '2005']
co2_emission_data = df_countries[['Country Name'] + years]

# Drop rows with missing values
co2_emission_data = co2_emission_data.dropna()

# Set 'Country Name' as the index
co2_emission_data.set_index('Country Name', inplace=True)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(co2_emission_data)

# Perform KMeans clustering
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(normalized_data)

# Add cluster labels to the DataFrame
co2_emission_data['Cluster'] = labels

# Visualize the clusters
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Cluster for 1999
axs[0].scatter(co2_emission_data[years[0]], co2_emission_data.index, c=labels, cmap='viridis')
axs[0].set_title(f'Clustered CO2 Emission in {years[0]}')
axs[0].set_xlabel('CO2 Emission')
axs[0].set_ylabel('Countries')

# Cluster for 2005
axs[1].scatter(co2_emission_data[years[1]], co2_emission_data.index, c=labels, cmap='viridis')
axs[1].set_title(f'Agriculture La in {years[1]}')
axs[1].set_xlabel('CO2 Emission')
axs[1].set_ylabel('Countries')

# Manually set y-axis label
for ax in axs:
    ax.set_yticks([])
    ax.set_yticklabels([])

plt.tight_layout()
plt.show()

co2_emission_data.head()