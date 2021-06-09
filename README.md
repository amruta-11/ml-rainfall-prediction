# Rainfall Prediction using Machince Learning Algorithms
This project is a part of Data Science II: Machine Learning course taught by Prof. Chirag Shah, in which I collaborated with 4 of my classmates.

### Problem Statement:
In this weather forecasting problem, we are focusing on following two predictions for the weekend after the final presentation (13 and 14 the of the March):
1. Will it rain or not?
2. How much will it rain?  

And, depending upon the results of our model, we wish to plan the spring break vacation. We scoped our project to predict the rainfall in the 6 popular vacation destinations i.e., Los Angeles,  Miami, Honolulu, San Diego, Yellowstone National Park and Boston. My task is to work on Los Angeles city. 

<p align="center"> 
<img src="https://github.com/amruta-11/ml-rainfall-prediction/blob/main/UScities.png">
</p>

### Data:
We collected the historical climate data for the cities of interest from the US National Oceanic and Atmospheric Administration (NOAA) website. The data files were in .dly format that needed to be converted into .csv files. We did this using a tool that we found on GitHub - [Get NOAA GHCN Data (Penne and Paulus, 2018)](https://github.com/aaronpenne/get_noaa_ghcn_data). This tool functions as an interface to the extensive weather data in the NOAA database.
[Here](https://github.com/amruta-11/ml-rainfall-prediction/blob/main/LA_historical_climate_dataset.csv) is an example data file (.csv) for LA city station.
The elements in the station data that we focused for this project are:
1.	YEAR - is the year of the record
2.	MONTH - is the month of the record
3.	DAY - is the day of the month
4.	PRCP - Precipitation (tenths of mm)
5.	TMAX - Maximum temperature (tenths of degrees C)
6.	TMIN - Minimum temperature (tenths of degrees C)

### ML Algorithms:
- The first question 'Will it rain or not?' was a classification problem and we selected logistic regression, k nearest neighbor (kNN) and random forest algorithms
- The second question 'How much will it rain?' was a regression problem and hence we selected linear regression

We built two models:
1. Basic Model: For the basic classification models (kNN, Logistic Regression, and Random Forest) two features were used which included the year and day of the data.
[Here](https://github.com/amruta-11/ml-rainfall-prediction/blob/main/LABasicModel.py) is the code for basic classification model for LA city.

2. Iterative Model: In iterated model, we decided to add some additional information to our list of independent variables. Specifically, TMIN, TMAX, and PRCP values from the past 5 days were included in each observation. This makes sense intuitively, since if the past few days have been getting colder and rainier, we’re likely to see rain today.
[Here](https://github.com/amruta-11/ml-rainfall-prediction/blob/main/LAIterative.py) is the code for iterated model for LA city.

After running the basic model we compared the accuracy for all three ML algorithms and the results were as below:
<p align="center"> 
<img src="https://github.com/amruta-11/ml-rainfall-prediction/blob/main/AccuracyforLogKNN%26RF.png">
</p>

We also compared the accuracies for the basic and iterative model for logistic regression algoritthm and the results were as below:
<p align="center"> 
<img src="https://github.com/amruta-11/ml-rainfall-prediction/blob/main/Comparison-of-Basic%26IterativeModel.png">
</p>

### Results:
Predictions:

<p align="center"> 
<img src="https://github.com/amruta-11/ml-rainfall-prediction/blob/main/results.png">
</p>
  
As seen in the figure above, logistic regression model predicted that it would not rain in any of the cities, except Seattle this weekend. In contrast however, our kNN model predicted it would rain both days in Boston, and one day in both Yellowstone and Seattle. Los Angeles, Miami, Honolulu, and San Diego were once again predicted to have no rainfall. The random forest model had the most mixed results. Here, neither of the locations predicted to have ‘no rain’ either day. Los Angeles, Yellowstone, and Seattle were predicted to have two days of rain, while Miami, Honolulu, Boston, and San Diego were predicted to have one day of rain
