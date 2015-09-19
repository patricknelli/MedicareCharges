# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 09:23:51 2015

@author: patrick.nelli
"""
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt

#importing cms 2013 charges dataset
charges = pd.read_csv('data//Medicare_Provider_Charge_Inpatient_DRG100_FY2013.csv')
charges.head()
charges.dtypes
charges.describe()
#substring out DRG Definition column to get the DRG number
charges['drg'] = charges['DRG Definition'].apply(lambda x: x[:4]).astype(int)
charges['DRG Definition'] = charges['DRG Definition'].apply(lambda x: x[6:])
charges.head()

#importing cms 2013 hospital compare hospital discription dataset
hospitalCol = ['Provider Number','County', 'Hospital Type','Hospital Ownership']
hospital = pd.read_csv('data//Hospital_Data.csv', usecols= hospitalCol)
hospital.head()
hospital.dtypes

#importing 2013 drg dataset that provides some additional information on each drg
drgCol = ['drg', 'glos', 'alos', 'postacute', 'mdc', 'medsurg', 'mdcdesc']
drg = pd.read_csv('data//drg2013.csv', usecols= drgCol)
drg.head()
drg.dtypes

#changing foreign keys to be strings so join will work
charges['Provider Id'] = charges['Provider Id'].astype(str)
charges.dtypes

#joining charges and hospital datasets to have additional information (e.g. hospital type) on each hospital
hospitalCharges = pd.merge(charges, hospital, left_on='Provider Id', right_on= 'Provider Number', how='left')
hospitalCharges.head()
hospitalCharges.describe()

#hospitals that didn't join between two data sets
hospitalsNotJoining = hospitalCharges['Provider Name'][hospitalCharges['Provider Number'].isnull()].unique()

#total discharges from these hospitals 
hospitalsNotJoiningDischarges = hospitalCharges['Total Discharges'][hospitalCharges['Provider Number'].isnull()].sum()

#% of overall total discharges
percentOfNotJoining = (float(hospitalsNotJoiningDischarges) / hospitalCharges['Total Discharges'].sum())*100 
print """The hospitals that don't join between the datasets account for %.2f percent 
    of the overall discharges""" % percentOfNotJoining
    
#joining with additional drg information to get full denormalized table
fullDF = pd.merge(hospitalCharges, drg, on='drg', how='left')
fullDF.head()
fullDF.describe()

#all drgs joined (function below equals 0)
fullDF['mdc'].isnull().sum()

#rearranging columns to group them in three general categories: hospital information, drg information, and charges information
cols = fullDF.columns.tolist()
cols = ['Provider Id',
 'Provider Name',
 'Provider Street Address',
 'Provider City',
 'County',
 'Provider State',
 'Provider Zip Code',
 'Hospital Referral Region (HRR) Description',
 'Hospital Type',
 'Hospital Ownership',
 'drg',
 'DRG Definition',
 'glos',
 'alos',
 'postacute',
 'mdc',
 'medsurg',
 'mdcdesc',
 'Total Discharges',
 'Average Covered Charges',
 'Average Total Payments',
 'Average Medicare Payments']
 
fullDF = fullDF[cols]
fullDF['Total Payments'] = fullDF['Total Discharges'] * fullDF['Average Total Payments']
fullDF.head()
fullDF.dtypes
fullDF.describe()

#pivot out DRGs so we can compare drgs to each other
pivotedDRG = fullDF.pivot_table(index=['Provider Id'],
                           columns=['drg','DRG Definition', 'mdc'],
                           values='Average Total Payments')
pivotedDRG.head()

#highest payment DRGs
pivotedDRG.mean().order(ascending = True).plot(kind='barh', figsize=[9, 25])
plt.title('Average Total Payments by DRG')
plt.ylabel('DRG')
plt.xlabel('Average Total Payment')
plt.show();

#highest variable DRGs (coeficient of variation)
coefficientOfVariation = pivotedDRG.std() / pivotedDRG.mean()
coefficientOfVariation.order(ascending = True).plot(kind='barh', figsize=[4, 25])
plt.title('Coefficient of Variation of Total Payments by DRG')
plt.ylabel('DRG')
plt.xlabel('Coefficient of Variation')
plt.show();



#process to create a bubble chart 
drgDF = fullDF.groupby(['drg','mdc'])
totalSpend = drgDF['Total Payments'].sum()
coefficientOfVariation = drgDF['Average Total Payments'].std() / drgDF['Average Total Payments'].mean()
totalCases = drgDF['Total Discharges'].sum()
mdc = pd.DataFrame(index = fullDF['mdc'].unique())

#create a dataframe to contain the bubble chart datapoints
drgScatterDF = pd.DataFrame()
drgScatterDF.head(50)

drgScatterDF['totalSpend'] = totalSpend
drgScatterDF['coefficientOfVariation'] = coefficientOfVariation
drgScatterDF['totalCases'] = totalCases
drgScatterDF.reset_index(inplace=True)
drgScatterDF.head()

# making the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(drgScatterDF.totalSpend, drgScatterDF.coefficientOfVariation
            , c=drgScatterDF.mdc, s= drgScatterDF['totalCases'] / 200
            , alpha = .5, label = drgScatterDF.mdc)
plt.figtext(drgScatterDF.totalSpend, drgScatterDF.coefficientOfVariation
            ,drgScatterDF.drg)
plt.title('KPA for DRG')
plt.xlabel('Total Payments per DRG')
plt.ylabel('Coefficient of Variation')
plt.annotate(drgScatterDF.drg.astype(str)
            , xy = (drgScatterDF.totalSpend, drgScatterDF.coefficientOfVariation))
plt.show();



#group DRGs by mdc
mdcDF = fullDF.groupby('mdcdesc')
mdcDF.describe()

#highest payment MDCs
mdcDF['Average Total Payments'].mean().order(ascending = True).plot(kind='barh', figsize=[3, 10])
plt.title('Average Total Payments by MDC')
plt.ylabel('MDC')
plt.xlabel('Average Total Payment')
plt.show();

#highest variable MDCs (coeficient of variation)
coefficientOfVariation = mdcDF['Average Total Payments'].std() / mdcDF['Average Total Payments'].mean()
coefficientOfVariation.order(ascending = True).plot(kind='barh', figsize=[3, 10])
plt.title('Coefficient of Variation of Total Payments by MDC')
plt.ylabel('MDC')
plt.xlabel('Coefficient of Variation')
plt.show();

