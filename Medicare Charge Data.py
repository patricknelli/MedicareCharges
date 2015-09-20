# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 09:23:51 2015

@author: patrick.nelli
"""
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import linregress

#not sure what the below items do, but they are listed 
#on the Seaborn site as intro lines of code
sns.set()
np.random.seed(sum(map(ord, "palettes")))



######  IMPORT AND CREATING DENORMALIZED PULL DATASET SECTION######

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
hospitalCharges = pd.merge(charges, hospital, left_on='Provider Id'
    , right_on= 'Provider Number', how='left')
hospitalCharges.head()
hospitalCharges.describe()

#hospitals that didn't join between two data sets
hospitalsNotJoining = hospitalCharges['Provider Name'][hospitalCharges['Provider Number'].isnull()].unique()

#total discharges from these hospitals 
hospitalsNotJoiningDischarges = hospitalCharges['Total Discharges'][hospitalCharges['Provider Number'].isnull()].sum()

#% of overall total discharges
percentOfNotJoining = (float(hospitalsNotJoiningDischarges) / \
    hospitalCharges['Total Discharges'].sum())*100 
print """The hospitals that don't join between the datasets account for %.2f percent 
    of the overall discharges""" % percentOfNotJoining
    
#joining with additional drg information to get full denormalized table
fullDF = pd.merge(hospitalCharges, drg, on='drg', how='left')
fullDF.head()
fullDF.describe()

#all drgs joined (function below equals 0)
fullDF['mdc'].isnull().sum()

#rearranging columns to group them in three general categories: ...
#...hospital information, drg information, and charges information
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

fullDF['Hospital Ownership'].fillna('NA', inplace=True)

#function to create a grouping of hospital ownerships
def convertOwnership(df):
    b = df['Provider Id']
    if df['Hospital Ownership'][:3] == 'Gov':
        a = 'Government'
    elif df['Hospital Ownership'][:3] == 'Vol':
        a = 'NonProfit'
    elif df['Hospital Ownership'] == 'NA':
        a = 'Unknown'
    else:
        a = 'Proprietary'
    return pd.Series(dict(OwnershipGroup=a, ProviderId=b)) 

ownershipGroup = fullDF.apply(convertOwnership, axis=1)

fullDF = pd.concat([fullDF, ownershipGroup], axis=1)
del fullDF['ProviderId']
fullDF.head()



######  EXPLORING DRGS SECTION  ######

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

#process to create a bubble chart to show highly variable and high payment DRGs
drgDF = fullDF.groupby(['drg','mdc'])
totalSpend = drgDF['Total Payments'].sum()
coefficientOfVariation = drgDF['Average Total Payments'].std() / \
    drgDF['Average Total Payments'].mean()
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

# making the scatter plot / bubble chart
plt.figure(figsize=(10, 10))
plt.scatter(drgScatterDF.totalSpend, drgScatterDF.coefficientOfVariation
            , c=drgScatterDF.mdc, s= drgScatterDF['totalCases'] / 200
            , alpha = .5)
for k, v in drgScatterDF[['totalSpend', 'coefficientOfVariation']].iterrows():
    plt.annotate(drgScatterDF['drg'].ix[k], v,
                xytext=(-10,-4), textcoords='offset points',
                family='sans-serif', fontsize=12, color='darkslategrey')
plt.title('KPA for DRG')
plt.xlabel('Total Payments per DRG')
plt.ylabel('Coefficient of Variation')
plt.show();


######  EXPLORING MDCS SECTION  ######

#group DRGs by mdc
mdcDF = fullDF.groupby('mdcdesc')
mdcDF.describe()

#highest payment MDCs
mdcDF['Average Total Payments'].mean().order(ascending = True).plot(kind='barh'
    , figsize=[3, 10])
plt.title('Average Total Payments by MDC')
plt.ylabel('MDC')
plt.xlabel('Average Total Payment')
plt.show();

#highest variable MDCs (coeficient of variation)
coefficientOfVariation = mdcDF['Average Total Payments'].std() / \
    mdcDF['Average Total Payments'].mean()
coefficientOfVariation.order(ascending = True).plot(kind='barh', figsize=[3, 10])
plt.title('Coefficient of Variation of Total Payments by MDC')
plt.ylabel('MDC')
plt.xlabel('Coefficient of Variation')
plt.show();


######  EXPLORING DRG 871 SECTION SINCE IT IS ...
######  ...A HIGH PAYMENT DRG AS SEEN IN BUBBLE CHART  ######

#Exploring DRG 871 since it is a high payment and variable drg
fullDF871 = fullDF[fullDF['drg'] == 871]
fullDF871.head()
fullDF871.describe()
fullDF871.reset_index(inplace=True)

plt.scatter(fullDF871['Total Discharges'], fullDF871['Average Total Payments']
    , s = 1, alpha = 0.5)
plt.title('Total Discharges versus Average Total Payments')
plt.ylabel('Average Total Payments')
plt.xlabel('Total Discharges')
plt.show();

#how does onwership relate to payments based on discharges
sns.lmplot('Total Discharges'
            , 'Average Total Payments'
            , data=fullDF871
            , hue='OwnershipGroup'
            , fit_reg=False);

#same chart as above
pal = dict(Government="seagreen", Unknown="gray", NonProfit="blue", Proprietary="red")
g = sns.FacetGrid(fullDF871, hue="OwnershipGroup", palette=pal, size=5)
g.map(plt.scatter, "Total Discharges", "Average Total Payments", s=50
    , alpha=.2, linewidth=.5, edgecolor="white")
g.add_legend();

#testing out pair plots
g = sns.PairGrid(fullDF871, vars=["Average Total Payments", "Total Discharges"]
    , hue="OwnershipGroup")
g.map(plt.scatter);

#do hospitals differ on average total payments based on ownership?
plt.figure();
plt.hist([fullDF871['Average Total Payments'][fullDF871['OwnershipGroup'] == 'Government'],
          fullDF871['Average Total Payments'][fullDF871['OwnershipGroup'] == 'Proprietary'],
          fullDF871['Average Total Payments'][fullDF871['OwnershipGroup'] == 'NonProfit']
          ], label=['Government', 'Proprietary', 'NonProfit']
          , stacked=False,normed = True)
plt.legend()
plt.title('Average Total Payments by Hospital Ownership')
plt.ylabel('% of Total Hospitals in Bin')
plt.xlabel('Average Total Payments')
plt.show();

#similar chart to above
g = sns.FacetGrid(fullDF871, col="OwnershipGroup")
g.map(plt.hist, "Average Total Payments");

#attempt to graph all relevant parameters using seaborn
cols = ['Total Discharges',
 'Average Covered Charges',
 'Average Total Payments',
 'Average Medicare Payments',
 'OwnershipGroup']
 
subDF871 = fullDF871[cols]
subDF871.head()
subDF871.describe()

g = sns.PairGrid(subDF871, hue="OwnershipGroup")
g.map(plt.scatter);

#any time I try to map a histogram on the diagnol, it doesn't seem to work

#g = sns.PairGrid(subDF871.dropna())
#g.map_diag(sns.kdeplot, lw=3)
#g.map_diag(plt.hist)
#g.map_offdiag(plt.scatter);
#
#g = sns.pairplot(subDF871.dropna(), hue = 'OwnershipGroup', diag_kind="kde", size=2.5);
#
#g = sns.PairGrid(subDF871, hue="OwnershipGroup")
#g.map_diag(plt.hist)
#g.map_offdiag(plt.scatter)
#g.add_legend();
#
#g = sns.pairplot(subDF871, hue="OwnershipGroup", diag_kind="kde", size=2.5);



######  HOSPITAL MARKET SHARE SECTION BY HRR  ######

#determine marketshare per HRR
hospitalDF = fullDF.groupby(['Provider Id','Hospital Referral Region (HRR) Description'
    , 'OwnershipGroup'])
HRRDF = fullDF.groupby('Hospital Referral Region (HRR) Description')
totalDisHRR = pd.DataFrame(HRRDF['Total Discharges'].sum(), columns = ['Total Discharges'])
totalDisHRR.rename(columns={'Total Discharges': 'Total Discharges per HRR'}, inplace=True)
totalDisHRR.reset_index(inplace=True)
totalDisHRR.head()

hospitalDF2 = pd.DataFrame()

hospitalDF2['Average Total Payments'] = hospitalDF['Average Total Payments'].mean()
hospitalDF2['Total Discharges'] = hospitalDF['Total Discharges'].sum()
hospitalDF2.head()
hospitalDF2.reset_index(inplace=True)

hospitalDF2 = pd.merge(hospitalDF2, totalDisHRR
    , on='Hospital Referral Region (HRR) Description')

hospitalDF2.head(20)
hospitalDF2.describe()
hospitalDF2['HRR Market Share'] = hospitalDF2['Total Discharges'] / \
    hospitalDF2['Total Discharges per HRR']
hospitalDF2.head(10)
hospitalDF2.sort(['Total Discharges'], ascending = True).head(20)


#plot discharges by hospitals to see if we should remove smaller discharge hospitals
plt.hist(hospitalDF2['Total Discharges'][hospitalDF2['Total Discharges'] < 5000]
    , bins = 100)
plt.title('Total Discharges by Hospital')
plt.ylabel('Number of Hospitals in Bin')
plt.xlabel('Total Discharges Bin')
plt.show();

hospitalDF2['Total Discharges'][hospitalDF2['Total Discharges'] > 200].count() / \
    hospitalDF2['Total Discharges'].count().astype(float)

sns.lmplot('HRR Market Share'
            , 'Average Total Payments'
            , data=hospitalDF2[(hospitalDF2['Total Discharges'] > 500) & \
                (hospitalDF2['OwnershipGroup'] == 'Proprietary')]
            #, hue='OwnershipGroup'
            , fit_reg=False);

hospitalDFLT200 = hospitalDF2[hospitalDF2['Total Discharges'] > 500]

#Overall linear regression between HRR Market Share and Average Total Payments
print "ALL HOSPITALS linear regression between HRR Market Share and Average Total Payments"
slope, intercept, r_value, p_value, stderr_slope = linregress( \
    hospitalDFLT200['HRR Market Share'], hospitalDFLT200['Average Total Payments'])

print "slope:", slope
print "intercept:", intercept
print "r-squared:", r_value**2
print "p_value:", p_value
print '\n'

#low r-squared and low p value indicates that is a positive correlation but a lot of variability, not a lot of prediction power

#Linear regression based on ownership
for row in hospitalDFLT200['OwnershipGroup'].unique():
    slope, intercept, r_value, p_value, stderr_slope = linregress( \
        hospitalDFLT200['HRR Market Share'][hospitalDFLT200['OwnershipGroup'] == row]
        , hospitalDFLT200['Average Total Payments'][hospitalDFLT200['OwnershipGroup'] == row])
    print row.upper() ,  """Owned Hospitals -- Linear regression between 
        HRR Market Share and Average Total Payments"""
    print "slope:", slope
    print "intercept:", intercept
    print "r-squared:", r_value**2
    print "p_value:", p_value
    print '\n'
