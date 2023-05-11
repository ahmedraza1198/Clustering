
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import itertools as iter
import scipy.optimize as opt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as slhte_score
import warnings as wrn
wrn.filterwarnings(action='ignore')

dpi_1 = 140
dpi_2 = 600
fw = "bold"
fs = 8
red = "green"
green = "red"
purple = "orange"
orng = "purple"
ylw = "blue"
#  2 Indicators
# 1.Population
# 2.GDP
# file is about the population
file_1 = 'total population.csv'
# this file is about the GDP
file_2 = "GDP.csv"


def read_file(file_name=None):
    """Desc:
    Taking the csv format file and returning 2 data frame .First one is  original dataframe
     and second one is its transposed version.
    
    Parameters:
        file_name: Filename of csv format.
    Returns:
        [DataFrame, DataFrame Transposed]: The original dataframe
        and its transposed version."""
    df = pd.read_csv(file_name, skiprows=3)
    df.drop(df.columns[[1, 2, 3]], axis=1, inplace=True)
    return df, df.transpose()


# reading the Total Population file where population is the original format and 
# populationT is the transposed format of original data frame
population, populationT = read_file(file_1)

# the selected years clustering analysis
year_81 = '1981'
year_21 = '2021'

# listing regional countries and outliers that may distort in data
interest_reg_list = ['Aruba', 'Afghanistan', 'Angola', 'Albania', 'Andorra',
                     'United Arab Emirates', 'Argentina', 'Armenia', 'American Samoa',
                     'Antigua and Barbuda', 'Australia', 'Austria', 'Azerbaijan',
                     'Burundi', 'Belgium', 'Benin', 'Burkina Faso', 'Bangladesh',
                     'Bulgaria', 'Bahrain', 'Bahamas, The', 'Bosnia and Herzegovina',
                     'Belarus', 'Belize', 'Bermuda', 'Bolivia', 'Brazil', 'Barbados',
                     'Brunei Darussalam', 'Bhutan', 'Botswana', 'Canada', 'Switzerland',
                     'Channel Islands', 'Chile', 'China', "Cote d'Ivoire", 'Cameroon',
                     'Congo, Rep.', 'Colombia', 'Comoros', 'Cabo Verde', 'Costa Rica',
                     'Cuba', 'Curacao', 'Cayman Islands', 'Cyprus', 'Czechia',
                     'Djibouti', 'Dominica', 'Denmark', 'Dominican Republic', 'Algeria',
                     'Ecuador', 'Egypt, Arab Rep.', 'Eritrea', 'Spain', 'Estonia',
                     'Ethiopia', 'Finland', 'Fiji', 'France', 'Faroe Islands',
                     'Micronesia, Fed. Sts.', 'Gabon', 'United Kingdom', 'Georgia',
                     'Ghana', 'Gibraltar', 'Guinea', 'Gambia, The', 'Guinea-Bissau',
                     'Equatorial Guinea', 'Greece', 'Grenada', 'Greenland', 'Guatemala',
                     'Guam', 'Guyana', 'Hong Kong SAR, China', 'Honduras', 'Croatia',
                     'Haiti', 'Hungary', 'Indonesia', 'Isle of Man', 'India', 'Ireland',
                     'Iran, Islamic Rep.', 'Iraq', 'Iceland', 'Israel', 'Italy',
                     'Jamaica', 'Jordan', 'Japan', 'Kazakhstan', 'Kenya',
                     'Kyrgyz Republic', 'Cambodia', 'Kiribati', 'St. Kitts and Nevis',
                     'Korea, Rep.', 'Kuwait', 'Lao PDR', 'Lebanon', 'Liberia', 'Libya',
                     'St. Lucia', 'Liechtenstein', 'Sri Lanka', 'Lesotho', 'Lithuania',
                     'Luxembourg', 'Latvia', 'Macao SAR, China',
                     'St. Martin (French part)', 'Morocco', 'Monaco', 'Moldova',
                     'Madagascar', 'Maldives', 'Mexico', 'Marshall Islands',
                     'North Macedonia', 'Mali', 'Malta', 'Myanmar', 'Montenegro',
                     'Mongolia', 'Northern Mariana Islands', 'Mozambique', 'Mauritania',
                     'Mauritius', 'Malawi', 'Malaysia', 'Namibia', 'New Caledonia',
                     'Niger', 'Nigeria', 'Nicaragua', 'Netherlands', 'Norway', 'Nepal',
                     'Nauru', 'New Zealand', 'Oman', 'Pakistan', 'Panama', 'Peru',
                     'Philippines', 'Palau', 'Papua New Guinea', 'Poland',
                     'Puerto Rico', "Korea, Dem. People's Rep.", 'Portugal', 'Paraguay',
                     'Qatar', 'Romania', 'Russian Federation', 'Rwanda', 'Saudi Arabia',
                     'Sudan', 'Senegal', 'Singapore', 'Solomon Islands', 'Sierra Leone',
                     'El Salvador', 'San Marino', 'Somalia', 'Serbia', 'South Sudan',
                     'Sao Tome and Principe', 'Suriname', 'Slovak Republic', 'Slovenia',
                     'Sweden', 'Eswatini', 'Sint Maarten (Dutch part)', 'Seychelles',
                     'Syrian Arab Republic', 'Turks and Caicos Islands', 'Chad', 'Togo',
                     'Thailand', 'Tajikistan', 'Turkmenistan', 'Timor-Leste', 'Tonga',
                     'Trinidad and Tobago', 'Tunisia', 'Turkiye', 'Tuvalu', 'Tanzania',
                     'Uganda', 'Ukraine', 'Uruguay', 'United States', 'Uzbekistan',
                     'St. Vincent and the Grenadines', 'Venezuela, RB',
                     'British Virgin Islands', 'Virgin Islands (U.S.)', 'Vietnam',
                     'Vanuatu', 'Samoa', 'Kosovo', 'Yemen, Rep.', 'South Africa',
                     'Zambia', 'Zimbabwe']
# extract the required data for the clustering and drop missing values
pop_cluster = population[population['Country Name'].isin(interest_reg_list)][
    ['Country Name', year_81, year_21]].dropna()

# visualising data
plt.figure(dpi=140)
pop_cluster.plot(year_81, year_21, kind='scatter', color='red')
plt.title('Total Population')
plt.show()

# convert the dataframe to an array
pop = pop_cluster[[year_81, year_21]].values

max_value = pop.max()
min_value = pop.min()
print("Scaling---------")
# normalizing data of 2 years
scale_data = (pop - min_value) / (max_value - min_value)

print('Scaled------= ')

sse = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=330, n_init=13, random_state=23)
    kmeans.fit(scale_data)
    sse.append(kmeans.inertia_)

# plotting to check for appropriate number of clusters using elbow method
plt.style.use('seaborn')
plt.figure(dpi=dpi_1)
plt.plot(range(1, 11), sse, color=ylw)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.savefig('cluster_graph.png')
plt.show()

# finding the Kmeans clusters
ncluster = 2
kmeans = KMeans(n_clusters=ncluster, init='k-means++', max_iter=340, n_init=40,
                random_state=40)

# Fit the model to the data of 1981 and 2021 years
kmeans.fit(scale_data)

# labels
labels = kmeans.labels_

# Use the silhouette score to evaluate the quality of the clusters
print(f'Silhouette Score: {slhte_score(scale_data, labels)}')

# Extracting clusters centers
cent = kmeans.cluster_centers_
print(cent)

# Plot the scatter plot of the clusters
plt.style.use('seaborn')
plt.figure(dpi=dpi_1)
plt.scatter(scale_data[:, 0], scale_data[:, 1], c=kmeans.labels_)
plt.title('K-Means Clustering')
plt.xlabel('1981')
plt.ylabel('2021')
plt.show()

# Getting the Kmeans
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=313, n_init=12,
                random_state=80)
y_predict = kmeans.fit_predict(scale_data)
print(y_predict)

# creating new dataframe with the labels for each country
pop_cluster['cluster'] = y_predict
pop_cluster.to_csv('results.csv', index=False)

# plotting the normalised Population
plt.style.use('seaborn')
plt.figure(dpi=dpi_1)
plt.scatter(scale_data[y_predict == 0, 0], scale_data[y_predict ==
                                                      0, 1], s=44, c=green, label='cluster 0')
plt.scatter(scale_data[y_predict == 1, 0], scale_data[y_predict ==
                                                      1, 1], s=44, c=ylw, label='cluster 1')

plt.scatter(cent[:, 0], cent[:, 1], s=44, c=red, label='Centroids')
plt.title('Total Population (Scaled)', fontweight=fw)
plt.xlabel('1981', fontweight=fw)
plt.ylabel('2021', fontweight=fw)
plt.legend()
plt.show()

# converting centroid to its unnormalized form
cent = cent * (max_value - min_value) + min_value

# plotting the Population in their clusters with the centroid points
plt.figure(dpi=dpi_2)
plt.scatter(pop[y_predict == 0, 0], pop[y_predict == 0, 1],
            s=43, c=ylw, label='Below 0.2Billion')
plt.scatter(pop[y_predict == 1, 0], pop[y_predict == 1, 1],
            s=43, c=red, label='Above 0.2Billion')

plt.scatter(cent[:, 0], cent[:, 1], s=43, c=green, label='Centroids')
plt.title('Total Population', fontweight=fw, fontsize=fs)
plt.xlabel('1981', fontweight=fw, fontsize=fs)
plt.ylabel('2021', fontweight=fw, fontsize=fs)
plt.legend()
plt.show()

# reading the Total population file from the world bank format
population, population2 = read_file(file_1)

# rename the transposed data columns for population
final_df = population2.rename(columns=population2.iloc[0])
# drop the country name
final_df = final_df.drop(index=final_df.index[0], axis=0)
final_df['Year'] = final_df.index

# # fitting the for Colombia's population data
df_clmbia = final_df[['Year', 'Colombia']].apply(pd.to_numeric,
                                                 errors='coerce')
df_clmbia = df_clmbia.dropna()

plt.figure(figsize=(18, 8))
df_clmbia['Colombia'].plot(
    kind='bar',
    color=green
)
plt.title('Total population in Colombia Year wise')
plt.show()


# Using the Logistic function for curve fitting and forecasting the Total Population
def logistic_func(t, n0_pop, g_rate, t0):
    """ Calculating the logistic growth of a population of different coutries.
    Parameters:
        t: The current time.
        n0_pop: The initial population.
        g_rate: The growth rate.
        t0: The inflection point.
        
    Returns:
       The population at the given time
    and growth rate g"""

    f = n0_pop / (1 + np.exp(-g_rate * (t - t0)))
    return f


# Doing Error ranges calculation
def error_ranges(x0, func, params, sigma):
    # Upper and Lower limits
    lower_limit = func(x0, *params)
    upper_limit = lower_limit

    #     Preparing  upper and lower limits for parameters by creating the list of tuples
    up_low_list = []
    for para, sgma in zip(params, sigma):
        p_min = para - sgma
        p_max = para + sgma
        up_low_list.append((p_min, p_max))

    p_mix = list(iter.product(*up_low_list))

    # calculate the upper and lower limits
    for p in p_mix:
        y = func(x0, *p)
        lower_limit = np.minimum(lower_limit, y)
        upper_limit = np.maximum(upper_limit, y)

    return lower_limit, upper_limit


# fits the logistic data
param_clmbia, covar_clmbia = opt.curve_fit(logistic_func, df_clmbia['Year'], df_clmbia['Colombia'],
                                           p0=(3e12, 0.03, 2041))

# calculating the standard deviation
sigma_clmbia = np.sqrt(np.diag(covar_clmbia))

# creating a new column with the fit data
df_clmbia['fit'] = logistic_func(df_clmbia['Year'], *param_clmbia)

# Forecast for the next 20 years
year = np.arange(1960, 2041)
forecast_clmbia = logistic_func(year, *param_clmbia)

# calculates the error ranges
low_clmbia, up_clmbia = error_ranges(year, logistic_func, param_clmbia, sigma_clmbia)

# plotting China's Total Population
plt.figure(dpi=dpi_2)
plt.plot(df_clmbia["Year"], df_clmbia["Colombia"],
         label="Population", c='purple')
plt.plot(year, forecast_clmbia, label="Forecast", c=red)
plt.fill_between(year, low_clmbia, up_clmbia, color=purple,
                 alpha=0.7, label='Confidence Range')
plt.xlabel("Year", fontweight=fw, fontsize=fs)
plt.ylabel("Population", fontweight=fw, fontsize=fs)
plt.legend()
plt.title('Colombia', fontweight='bold', fontsize=fs)
plt.show()

# prints the error ranges
print(error_ranges(2041, logistic_func, param_clmbia, sigma_clmbia))

# fitting the Australia's Population data
df_aus = final_df[['Year', 'Australia']].apply(pd.to_numeric,
                                               errors='coerce')
df_aus = df_aus.dropna()

# fits the Australia logistic data
param_aus, covar_aus = opt.curve_fit(logistic_func, df_aus['Year'], df_aus['Australia'],
                                     p0=(3e12, 0.03, 2041))

# calculates the standard deviation for Australia data
sigma_aus = np.sqrt(np.diag(covar_aus))

# Forecast for the next 20 years
forecast_aus = logistic_func(year, *param_aus)

# calculate error ranges
low_aus, up_aus = error_ranges(year, logistic_func, param_aus, sigma_aus)

# plotting Australia Total Population
plt.style.use('seaborn')
plt.figure(dpi=dpi_2)
plt.plot(df_aus["Year"], df_aus["Australia"],
         label="Population")
plt.plot(year, forecast_aus, label="Forecast", c=green)
plt.fill_between(year, low_aus, up_aus, color=purple,
                 alpha=0.6, label="Confidence Range")
plt.xlabel("Year", fontweight=fw, fontsize=fs)
plt.ylabel("Population", fontweight=fw, fontsize=fs)
plt.legend(loc='upper left')
plt.title('Australia Population', fontweight=fw, fontsize=fs)
plt.show()

plt.figure(figsize=(17, 10))

df_aus['Australia'].plot(
    kind='bar',
    color=red
)
plt.title('Total population in Australia Year Wise')
plt.show()

# fitting the data
df_cnda = final_df[['Year', 'Canada']].apply(pd.to_numeric,
                                             errors='coerce')
df_cnda = df_cnda.dropna()

# fits the Ghana's logistic data
param_cnda, covar_cnda = opt.curve_fit(logistic_func, df_cnda['Year'], df_cnda['Canada'],
                                       p0=(3e12, 0.03, 2041))

# sigma is the standard deviation
sigma_cnda = np.sqrt(np.diag(covar_cnda))

# Forecast for the next 20 years
forecast_cnda = logistic_func(year, *param_cnda)

# calculate error ranges
low_cnda, up_cnda = error_ranges(year, logistic_func, param_cnda, sigma_cnda)

# creates a new column for the fit data
df_cnda['fit'] = logistic_func(df_cnda['Year'], *param_cnda)

# plotting Canada's Total Population
plt.figure(dpi=dpi_2)
plt.plot(df_cnda["Year"], df_cnda["Canada"],
         label="Population", c=red)
plt.plot(year, forecast_cnda, label="Forecast", c=green)
plt.fill_between(year, low_cnda, up_cnda, color=purple,
                 alpha=0.7, label='Confidence Range')
plt.xlabel("Year", fontweight=fw, fontsize=fs)
plt.ylabel("Population", fontweight=fw, fontsize=fs)
plt.legend(loc='upper left')
plt.title('Canada', fontweight=fw, fontsize=fs)
plt.show()

plt.figure(figsize=(16, 9))

df_cnda['Canada'].dropna().plot(
    kind='bar',
    color=green
)
plt.title('Total population in  Canada Year wise')
plt.show()

# reading the GDP/Capita file from the world bank format
gdp, gdpT = read_file(file_2)

# rename the columns
gdp = gdpT.rename(columns=gdpT.iloc[0])

# drop the country name
gdp = gdp.drop(index=gdp.index[0], axis=0)
gdp['Year'] = gdp.index

# fitting the data
gdp_cnda = gdp[['Year', 'Canada']].apply(pd.to_numeric,
                                         errors='coerce')
gdp_cnda = gdp_cnda.dropna()


# poly function for forecasting GDP/Capita
def poly(x, a, b, c):
    """ Calculates the value of a polynomial function of the form ax^2 + bx + c.
    
    Parameters:
        x: The input value for the polynomial function.
        a: The coefficient of x^2 in the polynomial.
        b: The coefficient of x in the polynomial.
        c: The constant term in the polynomial.
        
    Returns:
           The value of the polynomial function at x.
      
    """
    return a * x ** 2 + b * x + c


def get_error_estimates(x, y, degree):
    """
   Calculates the error estimates of a polynomial function.
   
   Parameters:
       x : The x-values of the data points.
       y : The y-values of the data points.
       degree: The degree of the polynomial.
       
   Returns:
       The standard deviation of the residuals as the error estimate.
       """

    coefficients = np.polyfit(x, y, degree)
    y_estimate = np.polyval(coefficients, x)
    residuals = y - y_estimate
    #  The standard deviation of the residuals as the error estimate
    return np.std(residuals)


# fits the linear data of Canada
param_cnda, cov_cnda = opt.curve_fit(poly, gdp_cnda['Year'], gdp_cnda['Canada'])

# calculates the standard deviation of Canada
sigma_cnda = np.sqrt(np.diag(cov_cnda))

# creates a new column for the fit figures into Canada
gdp_cnda['fit'] = poly(gdp_cnda['Year'], *param_cnda)

# forecasting the fit figures of Canada data
forecast_cnda = poly(year, *param_cnda)

# error estimates of Canada data
error_cnda = get_error_estimates(gdp_cnda['Canada'], gdp_cnda['fit'], 2)
print('\n Error Estimates for Canada GDP/Capita:\n', error_cnda)

# Plotting
plt.style.use('seaborn')
plt.figure(dpi=dpi_2)
plt.plot(gdp_cnda["Year"], gdp_cnda["Canada"],
         label="GDP/Capita", c=orng)
plt.plot(year, forecast_cnda, label="Forecast", c=green)

plt.xlabel("Year(Canada)", fontweight=fw, fontsize=fs)
plt.ylabel("Population(Canada)", fontweight=fw, fontsize=fs)
plt.legend()
plt.title('Canada', fontweight=fw, fontsize=fs)
plt.show()

# fitting the data
gdp_aus = gdp[['Year', 'Australia']].apply(pd.to_numeric,
                                           errors='coerce')
gdp_aus = gdp_aus.dropna()
# fits the linear data of Australia
param_aus, cov_aus = opt.curve_fit(poly, gdp_aus['Year'], gdp_aus['Australia'])

# calculates the standard deviation of Australia data
sigma_aus = np.sqrt(np.diag(cov_aus))

# creates a column for the fit data of Australia
gdp_aus['fit'] = poly(gdp_aus['Year'], *param_aus)

# forecasting for the next 20 years of Australia
forecast_aus = poly(year, *param_aus)

# error estimates of Australia
error_aus = get_error_estimates(gdp_aus['Australia'], gdp_aus['fit'], 2)
print('\n Error Estimates for US GDP/Capita:\n', error_aus)

# plotting
plt.figure(dpi=dpi_2)
plt.plot(gdp_aus["Year"], gdp_aus["Australia"],
         label="GDP/Capita")
plt.plot(year, forecast_aus, label="Forecast", c='red')
plt.xlabel("Year", fontweight='bold', fontsize=14)
plt.ylabel("GDP per Capita ('Australia$')", fontweight='bold', fontsize=14)
plt.legend()
plt.title('Australia', fontweight='bold', fontsize=14)
plt.show()

# fitting the data of Colombia
gdp_clmbia = gdp[['Year', 'Colombia']].apply(pd.to_numeric,
                                             errors='coerce')
gdp_clmbia = gdp_clmbia.dropna()
# fits the linear data of Colombia
param_clmbia, cov_clmbia = opt.curve_fit(poly, gdp_clmbia['Year'], gdp_clmbia['Colombia'])

# calculates the standard deviation of Colombia
sigma_clmbia = np.sqrt(np.diag(cov_clmbia))

# creates a new column for the fit data of Colombia
gdp_clmbia['fit'] = poly(gdp_clmbia['Year'], *param_clmbia)

# forescast paramaters for the next 20 years
forecast_clmbia = poly(year, *param_clmbia)

# error estimates
error_clmbia = get_error_estimates(gdp_clmbia['Colombia'], gdp_clmbia['fit'], 2)
print('\n Error Estimates for Colombia GDP/Capita:\n', error_clmbia)

# plotting
plt.figure(dpi=dpi_2)
plt.plot(gdp_clmbia["Year"], gdp_clmbia["Colombia"],
         label="GDP/Capita", c=red)
plt.plot(year, forecast_clmbia, label="Forecast", c=green)
plt.xlabel("Year", fontweight=fw, fontsize=fs)
plt.ylabel("GDP per Capita ('Colombia$')", fontweight=fw, fontsize=fs)
plt.legend()
plt.title('Colombia', fontweight=fw, fontsize=fs)
plt.show()
