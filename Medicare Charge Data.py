# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 09:23:51 2015

@author: patrick.nelli
"""
import numpy as np
from pandas import Series, DataFrame
import pandas as pd

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
fullDS = pd.merge(hospitalCharges, drg, on='drg', how='left')
fullDS.head()
fullDS.describe()

#all drgs joined (function below equals 0)
fullDS['mdc'].isnull().sum()

#rearranging columns to group them in three general categories: hospital information, drg information, and charges information
cols = fullDS.columns.tolist()
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
 
fullDS = fullDS[cols]
fullDS.head()
fullDS.dtypes
fullDS.describe()





