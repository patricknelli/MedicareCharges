# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 09:23:51 2015

@author: patrick.nelli
"""
import numpy as np
from pandas import Series, DataFrame
import pandas as pd

charges = pd.read_csv('data//Medicare_Provider_Charge_Inpatient_DRG100_FY2013.csv')
charges.head()
charges.dtypes
charges.describe()

hospitalCol = ['Provider Number','County', 'Hospital Type','Hospital Ownership']
hospital = pd.read_csv('data//Hospital_Data.csv', usecols= hospitalCol)
hospital.head()
hospital.dtypes

drgCol = ['drg', 'glos', 'alos', 'postacute', 'mdc', 'medsurg', 'mdcdesc']
drg = pd.read_csv('data//drg2013.csv', usecols= drgCol)
drg.head()
drg.dtypes
