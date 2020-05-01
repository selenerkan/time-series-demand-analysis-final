## Project: Time Series Demand Analysis
## Table of contents
* [Install](#Install)
* [CODE](#CODE)
* [RUN](#RUN)
* [EXAMPLE OUTPUT](#EXAMPLE-OUTPUT)

## Install
This project requires Python and the following Python libraries installed:
    
    • NumPy
    • Pandas    
    • matplotlib
    • sklearn
    • pyearth
    • json
    • webbrowser
    • tkinter
    • os
    • time
    • seaborn
    • statsmodels
    • math
    • sys
    • warnings
    • itertools
You need to have Python 3.6 in order to run the program and install the required libraries. If you face any problems regarding the installation of the pyearth library, you may try to use earlier Python versions (2.6). 
	
## CODE
You need to install every .py files and the src folder. Then, you need to run main.py file. 
	
## RUN
After you run the main.py file, there will be a GUI page shown as below.

![image](https://user-images.githubusercontent.com/56449035/80843423-f409ca80-8c0c-11ea-89bc-dd7dd82faf4c.png)

 
You need to select the data file that you want to use. If you want to apply forecasting, while the target columns is empty for the forecasting period, other features must be filled. While testing, target column cannot be empty. So, you shouldn’t include the forecasting period data into your dataset. The data that you will use must have a timestamp (date) column and must be a .csv file. Timestamp column must be in the format of DD-MM-YYYY. You also need to specify the index numbers of the target and the timestamp column (Indices start from 0). Timestamp column’s index number should be less than the index of the target column. 

After completing these steps, you need to choose whether you’d like to apply PCA or feature selection.

If you choose to apply PCA: In the “enter total feature number after PCA” section, you need to specify the dimension that you want to reduce to.

If you choose to apply Feature Selection: In the “enter total feature number after PCA” section, you need to enter “0”. 

In the testing or forecasting part, we advise you first test all of the algorithms on your dataset and based on the output report, you can choose the algorithm(s) that you want to run and apply forecasting. 

If you choose to test algorithm(s): Enter the percentage of the training data in the “enter the index to start forecasting section”. (Ex. 80)

If you choose to forecast: You have to enter the row index that you want to start your forecast from to “enter the index to start forecasting section” section (ex. 380, it will do the forecasting starting from row 381. Indices start from 0). While the target column is not required, other features should be included. 

In the right hand side of the GUI page, you can see the listed classification and regression algorithm. You can choose one or more algorithms based on the target feature type. (If you will use the regression algorithms, check the “conduct regression” box)

Then you can click the run button. Example filled GUI page is provided below:

![image](https://user-images.githubusercontent.com/56449035/80843532-38956600-8c0d-11ea-9a12-5f6cd6be52eb.png)

## EXAMPLE OUTPUT


![image](https://user-images.githubusercontent.com/56449035/80843883-059fa200-8c0e-11ea-93ed-93bf72b9ed88.png)

![image](https://user-images.githubusercontent.com/56449035/80843956-341d7d00-8c0e-11ea-82ba-b93dea6d80d5.png)
