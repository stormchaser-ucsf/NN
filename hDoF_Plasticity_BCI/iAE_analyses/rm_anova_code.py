# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:03:36 2022

@author: nikic
"""

import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
import mat73

# df = pd.DataFrame({'patient': np.repeat([1, 2, 3, 4, 5], 4),
#                    'drug': np.tile([1, 2, 3, 4], 5),
#                    'response': [30, 28, 16, 34,
#                                 14, 18, 10, 22,
#                                 24, 20, 18, 30,
#                                 38, 34, 20, 44, 
#                                 26, 28, 14, 30]})

filename = 'C:/Users/nikic/Documents/Ganguly lab/EEG Sleep Ana/data_rm.mat'
data_dict = mat73.loadmat(filename)
tmpp = data_dict.get('data_rm')
tmpp = tmpp.T

df = pd.DataFrame({'subject': tmpp[:,0],
                   'time_pt': tmpp[:,1],
                   'stai_score':tmpp[:,2]})


print(AnovaRM(data=df, depvar='stai_score', subject='subject', within=['time_pt']).fit())


# performing a two way ANOVA on high CAPS
filename = 'C:/Users/nikic/Documents/Ganguly lab/EEG Sleep Ana/data_rm_2way.mat'
data_dict = mat73.loadmat(filename)
tmp = data_dict.get('data_rm_2way')
df = pd.DataFrame({'subject': tmp[:,0],                   
                   'stai_score':tmp[:,1],
                   'sleep_time':tmp[:,3],
                   'sleep_visit':tmp[:,2]})
rm2way = AnovaRM(df, depvar='stai_score', subject='subject',within=['sleep_time', 'sleep_visit'])
res2way = rm2way.fit()
print(res2way)


# performing a two way ANOVA on low CAPS
filename = 'C:/Users/nikic/Documents/Ganguly lab/EEG Sleep Ana/data_rm_2way_low.mat'
data_dict = mat73.loadmat(filename)
tmp = data_dict.get('data_rm_2way_low')
df = pd.DataFrame({'subject': tmp[:,0],                   
                   'stai_score':tmp[:,1],
                   'sleep_time':tmp[:,3],
                   'sleep_visit':tmp[:,2]})
rm2way = AnovaRM(df, depvar='stai_score', subject='subject',within=['sleep_time', 'sleep_visit'])
res2way = rm2way.fit()
print(res2way)


# performing an overall two ANOVA on all CAPS
filename = 'C:/Users/nikic/Documents/Ganguly lab/EEG Sleep Ana/data_rm_overall.mat'
data_dict = mat73.loadmat(filename)
tmp = data_dict.get('data_rm_overall')
df = pd.DataFrame({'subject': tmp[:,0],                   
                   'stai_score':tmp[:,1],
                   'sleep_time':tmp[:,3],
                   'sleep_visit':tmp[:,2]})
rm2way = AnovaRM(df, depvar='stai_score', subject='subject',within=['sleep_time', 'sleep_visit'])
res2way = rm2way.fit()
print(res2way)







