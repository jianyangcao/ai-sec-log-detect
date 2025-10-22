import pandas as pd

df=pd.read_csv('data/day_8')
print('loaded CSV sucessfully')

#insepct the datastructure
print(df.shape) #U(rows, columns)
print(df.info()) #data types and non-null counts
print(df.columns) #column names
print(df.head()) #print first 5 rows

#get basic statistics
print(df.describe()) #numeric summary
print(df['event'].value_counts()) #example cateforical count

#handle indexing and selection
events=df['event']#slect specific column
errors=df[df['status']=='error'] #filter rows
critial_logs=df[(df['status']=='warning') & (df['user']=='root')]

print(df.isnull().sum()) #count of missing per column

summary=df.describe(include='all') #summary for all data types
summary.to_csv('reports/day8_summary.csv')