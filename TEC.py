# MACHINE LEARNING REGRESSION MODELS FOR PREDICTION OF MULTIPLE IONOSPHERIC PARAMETERS

# Code for TEC parameter prediction

# PRELIMINARIES and PREPROCESSING 
#--------------------------------------------------------------------------------------------------------
# Importing libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.font_manager as font_manager

# Adjusting plot parameters
plt.rcParams.update({'font.size': 16})
plt.style.use('default')
csfont = {'fontname':'Times New Roman'}
hfont = {'fontname':'Times New Roman'}
COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR
font = font_manager.FontProperties(family='Times New Roman',weight='bold',style='italic', size=16)

# Reading the input file for training and validation
veri = pd.read_excel("roma_ion.xlsx")
veri

# Dropping hmF2 and f0F2 columns which we won't use for TEC predictions.
veri = veri.drop(['f0F2(t-23)', 'f0F2','hmF2(t-23)', 'hmF2' ], axis=1)

# Dropping rows including NaN values
veri_nonNaN = veri.dropna(axis=0, how='any')
veri_nonNaN.isnull().sum()
veri_nonNaN.shape
veri_nonNaN

# Re-indexing the Non-NaN dataframe
veri_nonNaN.reset_index(drop=True, inplace=True)

veri_nonNaN.shape

# Variable selection
X = np.asarray(veri_nonNaN[["DNS","DNC","HRS","HRC","f107","ap","TEC(t-23)"]]) # INDEPENDENT VARIABLES
y = np.asarray(veri_nonNaN["TEC"]) # DEPENDENT VARIABLE TO PREDICT

# MODELLING
#--------------------------------------------------------------------------------------------------------------------
# TRAIN-TEST SPLIT
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# DECISION TREE REGRESSOR AND ITS R SCORE ON TRAIN AND TEST DATA

from sklearn.tree import DecisionTreeRegressor

regrDT = DecisionTreeRegressor(max_depth=10)
regrDT.fit(X_train,y_train)
print("score train DT: ",regrDT.score(X_train,y_train))
print("score test DT: ",regrDT.score(X_test,y_test))

# SUPPORT VECTOR REGRESSOR AND ITS R SCORE ON TRAIN AND TEST DATA

from sklearn.svm import SVR
regrSVR = SVR()
regrSVR.fit(X_train,y_train)
print("score train SVR: ",regrSVR.score(X_train,y_train))
print("score test SVR: ",regrSVR.score(X_test,y_test))

# RANDOM FOREST REGRESSOR AND ITS R SCORE ON TRAIN AND TEST DATA

from sklearn.ensemble import RandomForestRegressor
regRF = RandomForestRegressor(n_estimators= 1000, random_state=100)
regRF.fit(X_train,y_train)
print("score train RF: ",regRF.score(X_train,y_train))
print("score test RF: ",regRF.score(X_test,y_test))

# PREDICTIONS FOR DECISION TREE REGRESSION

# WARNING: THE PREDICTIONS AND OTHER CALCULATIONS CAN BE ALSO DONE FOR RANDOM FOREST AND SUPPORT VECTOR MACHINE REGRESSORS
# BY CHANGING THE VARIABLE OF REGRESSOR (regrSVR AND regRF)
# YOU CAN ALSO CHANGE THE LABELS AND EXPORTED FILE NAMES DEPENDING ON YOUR REGRESSOR SELECTION.
# HERE, WE PRESENT ONLY DECISION TREE REGRESSION IN ORDER NOT TO SUBMIT A LONG AND COMPLEX CODE.
#--------------------------------------------------------------------------------------------------------------------

veri2 = pd.read_excel("roma_ion.xlsx",sheet_name="2013_predictdata")
# Dropping hmF2 and f0F2 columns which we won't use for TEC predictions.
veri2 = veri2.drop(['f0F2(t-23)', 'f0F2','hmF2(t-23)', 'hmF2' ], axis=1)
# Dropping rows including NaN values
veri2_nonNaN = veri2.dropna(axis=0, how='any')
veri2_nonNaN.isnull().sum()
veri2_nonNaN.shape
# Re-indexing the Non-NaN dataframe
veri2_nonNaN.reset_index(drop=True, inplace=True)
#Selection of columns to be predicted
X_validate = np.asarray(veri2_nonNaN[["DNS","DNC","HRS","HRC","f107","ap","TEC(t-23)"]])
y_validate = np.asarray(veri2_nonNaN["TEC"])
print(X_validate.shape)
print(y_validate.shape)

# Applying the regressor model
predictions = regrDT.predict(X_validate)

# Plotting the predictions and observed data of 2013 All Year

ticks = [0,728,1350,2088,2612,3284,4004,4746,5377,6041,6785,7422]
labels = ["Jan 01", "Feb 01", "Mar 01", "Apr 01", "May 01", "Jun 01", "Jul 01", "Aug 01", "Sep 01", "Oct 01", "Nov 01", "Dec 01"]

plt.figure(figsize=(18,5))
plt.plot(predictions,"yellow",label="Predictions in 2013")
plt.plot(y_validate,"blue", label="Observed Data in 2013", alpha = 0.5)
plt.title("Decision Tree Regression", pad=10, **csfont,style='italic', fontsize=16)
plt.xticks(ticks = ticks, labels = labels, **hfont,style='italic',fontsize=14)
plt.legend(bbox_to_anchor=(0.5,-0.22), frameon = False, loc="lower center",prop=font, ncol=2)
plt.ylabel("TEC (TECU)",labelpad=5, **hfont,style='italic',fontsize=14)
plt.yticks(**hfont,style='italic',fontsize=14)
plt.savefig('DT_AllYearPredictions_TEC.png', dpi=300)
plt.show()

# Accuracy Assessment for all year prediction
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_validate,predictions))
print("RMSE for All year predictions in 2013: " +"{:.4}".format(rmse))

# Segmenting the seasons
predictions_jan = predictions[0:728]
predictions_feb = predictions[728:1350]
predictions_mar = predictions[1350:2088]
predictions_apr = predictions[2088:2612]
predictions_may = predictions[2612:3284]
predictions_jun = predictions[3284:4004]
predictions_jul = predictions[4004:4746]
predictions_aug = predictions[4746:5377]
predictions_sep = predictions[5377:6041]
predictions_oct = predictions[6041:6785]
predictions_nov = predictions[6785:7422]
predictions_dec = predictions[7422:]
predictions_equinoxes = np.concatenate((predictions_mar,predictions_apr,predictions_sep,predictions_oct))
predictions_winter = np.concatenate((predictions_jan,predictions_feb,predictions_nov,predictions_dec))
predictions_summer = np.concatenate((predictions_may,predictions_jun,predictions_jul,predictions_aug))

y_validate_jan = y_validate[0:728]
y_validate_feb = y_validate[728:1350]
y_validate_mar = y_validate[1350:2088]
y_validate_apr = y_validate[2088:2612]
y_validate_may = y_validate[2612:3284]
y_validate_jun = y_validate[3284:4004]
y_validate_jul = y_validate[4004:4746]
y_validate_aug = y_validate[4746:5377]
y_validate_sep = y_validate[5377:6041]
y_validate_oct = y_validate[6041:6785]
y_validate_nov = y_validate[6785:7422]
y_validate_dec = y_validate[7422:]
y_validate_equinoxes = np.concatenate((y_validate_mar,y_validate_apr,y_validate_sep,y_validate_oct))
y_validate_winter = np.concatenate((y_validate_jan,y_validate_feb,y_validate_nov,y_validate_dec))
y_validate_summer = np.concatenate((y_validate_may,y_validate_jun,y_validate_jul,y_validate_aug))

# Accuracy Assessment for seasons prediction
rmse_equinoxes = np.sqrt(mean_squared_error(y_validate_equinoxes,predictions_equinoxes))
print("RMSE for equinoxes predictions in 2013: " +"{:.4}".format(rmse_equinoxes))
rmse_winter = np.sqrt(mean_squared_error(y_validate_winter,predictions_winter))
print("RMSE for winter predictions in 2013: " +"{:.4}".format(rmse_winter))
rmse_summer = np.sqrt(mean_squared_error(y_validate_summer,predictions_summer))
print("RMSE for summer predictions in 2013: " +"{:.4}".format(rmse_summer))

# Statistics
print(np.corrcoef(predictions,y_validate))
print(np.corrcoef(predictions_summer,y_validate_summer))
print(np.corrcoef(predictions_winter,y_validate_winter))
print(np.corrcoef(predictions_equinoxes,y_validate_equinoxes))
print(np.std(predictions))
print(np.std(predictions_summer))
print(np.std(predictions_winter))
print(np.std(predictions_equinoxes))
print(np.std(y_validate))
print(np.std(y_validate_summer))
print(np.std(y_validate_winter))
print(np.std(y_validate_equinoxes))

# Absolute Deviations for all year

# WARNING: YOU CAN OBTAIN THE SAME PLOTS FOR SEASONS OR MONTHS BY CHANGING THE INPUT VARIABLES
# HERE WE PRESENT THE PLOT ONLY FOR ALL YEAR PREDICTIONS 

dev_yil = np.abs(np.subtract(y_validate, predictions))
plt.figure(figsize=(18,5))
plt.plot(dev_yil ,"red",label="Deviations in 2013")
plt.ylabel("Absolute TEC Deviations",labelpad=5, **hfont,style='italic',fontsize=14)
plt.title("Decision Tree Regression", pad=10, **csfont,style='italic', fontsize=16)
ticks = [0,728,1350,2088,2612,3284,4004,4746,5377,6041,6785,7422]
labels = ["Jan 01", "Feb 01", "Mar 01", "Apr 01", "May 01", "Jun 01", "Jul 01", "Aug 01", "Sep 01", "Oct 01", "Nov 01", "Dec 01"]
y_ticks = [0,5,10,15,20,25,30]
plt.xticks(ticks = ticks, labels = labels, **hfont,style='italic',fontsize=14)
plt.yticks(ticks=y_ticks, **hfont,style='italic',fontsize=14)
plt.savefig('DT_AllYearDeviations_TEC', dpi=300)
plt.show()

# Histograms of Absolute Deviations

# WARNING: YOU CAN OBTAIN THE SAME PLOTS FOR SEASONS OR MONTHS BY CHANGING THE INPUT VARIABLES
# HERE WE PRESENT THE PLOT ONLY FOR ALL YEAR PREDICTIONS 

from matplotlib.ticker import PercentFormatter
bins= [0,2,4,6,8,10]
freq, bins, patches=plt.hist(dev_yil,bins=bins, weights=np.ones(len(dev_yil)) / len(dev_yil))

#plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xlabel("Absolute TEC Deviations", labelpad = 5,**hfont,style='italic',fontsize=14)
plt.ylabel("Frequency", labelpad=5,**hfont,style='italic',fontsize=14)
plt.title("Decision Tree Regression", pad = 10, **csfont,style='italic', fontsize=16)
plt.yticks(**hfont,style='italic',fontsize=14)
plt.xticks(**hfont,style='italic',fontsize=14)
counts, bins2, patches2 = plt.hist(dev_yil,bins=bins,edgecolor='black', color="blue")
# x coordinate for labels
bin_centers = np.diff(bins)*0.5 + bins[:-1]

n = 0
for fr, x, patch in zip(freq, bin_centers, patches):
  
    height = counts[n]
    freq_format = freq[n]
    plt.annotate("{:.2%}".format(freq_format),
               xy = (x, height),             # top left corner of the histogram bar
               xytext = (0.2,0.5),             # offsetting label position above its bar
               textcoords = "offset points", # Offset (in points) from the *xy* value
               ha = 'center', va = 'bottom',
               **hfont,style='italic',fontsize=10)
    n = n+1
plt.savefig('DT_AllYearDeviationsHist_TEC.png', dpi=300)

# Regression plots

# WARNING: YOU CAN OBTAIN THE SAME PLOTS FOR SEASONS OR MONTHS BY CHANGING THE INPUT VARIABLES
# HERE WE PRESENT THE PLOT ONLY FOR ALL YEAR PREDICTIONS 

plt.figure(figsize=(5,5))
plt.plot(predictions,y_validate, 'o', color = "blue")
m, b = np.polyfit(predictions, y_validate, 1)
plt.plot(predictions, m*predictions + b, label='y={:.2f}x+{:.2f}'.format(m,b), color="red")
plt.xlabel("TEC Predictions for 2013",  labelpad=5,**hfont,style='italic',fontsize=14)
plt.ylabel("TEC Observations in 2013", labelpad=5,**hfont,style='italic',fontsize=14)
plt.legend(loc = "lower right",prop=font)
plt.title("Decision Tree Regression", pad = 10, **csfont,style='italic', fontsize=16)
reg_xticks=[10,15,20,25,30,35,40,45,50,55,60]
reg_yticks=[10,15,20,25,30,35,40,45,50,55,60]
plt.yticks(ticks=reg_yticks,**hfont,style='italic',fontsize=14)
plt.xticks(ticks=reg_xticks,**hfont,style='italic',fontsize=14)
plt.savefig('DT_AllYearRegPlot_TEC.png', dpi=300)
plt.show()
