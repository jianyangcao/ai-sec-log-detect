import pandas as pd
from pathlib import Path

def clean_log(input_path: Path, output_path: Path):
    df=pd.read_csv(input_path)
    print('Before cleaning:',df.shape)

    #handle missing values
    df=df.dropna() #removes rows with Nan or using df.fillna(value="Unknown") replace missing values
    '''
    df.dropna(inplace=True) is equivalent to df=df.dropna()
    becuase inplace=True modefies df inside and returns None
    '''
    
    df.duplicated().sum() #check for duplicates
    df=df.drop_duplicates() #remove duplicates

    print('After cleaning:',df.shape)
    df.to_csv(output_path,index=False)
    print(f'Clearned file saved to {output_path}')

if __name__=='__main__':
        clean_log(Path('data/day9_sample.csv'), Path('reports/day9_cleaned_sample.csv'))