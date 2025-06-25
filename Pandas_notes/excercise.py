import numpy as np
import pandas as pd
ss=['y','n','y','n','n','y','y','n','n','y']
exam_data={
    'name':['a','a','s','r','l','m','r','k','b','c'],
    'score':[15.5,9,16.5,np.nan,9,50,17,np.nan,8,20],
    'attempts':[1,3,2,3,2,3,1,1,2,1],
    'qualify':['y','n','y','n','n','y','y','n','n','y']
}
labels = ["A","B",'C','D','E','F','G','H','I','J']

df = pd.DataFrame(exam_data,index=labels)
print(df['qualify'].describe())

df=df.reindex(columns=['score','qualify','new_column'],
              labels=['A','B','new_row'])
print(df.to_string())
