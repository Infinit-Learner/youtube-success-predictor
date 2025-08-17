import numpy as np
import pandas as pd
import json

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def merge_with_categories(category_json: dict , main_df: pd.DataFrame) -> pd.DataFrame:
    '''
   Combines main dataframe with the category title from category json 

   Args
    category_json (dict): json file with category id and category title feature
    main_df (pd.Dataframe): Main dataframe for category title to be add to

    Returns:
        pd.Dataframe: Combined dataframe 
    '''
    category_items = category_json['items']
    category_df = pd.DataFrame([{'category_id': int(item['id']), 
                                "category_title": item['snippet']['title']} for item in category_items])
    return main_df.merge(category_df, on = 'category_id', how = "left")

def run_preprocessing(raw_path: str, cat_id_path: str ,processed_path: str):
    '''
    Intakes a raw CSV from the raw_path and outputs a processed CSV to the processed path
    
    Args
        raw_path (str): The file path to the raw csv
        cat_id_path (str): File directory path to cateory id json file
        processed_path (str): The file path to output the processed csv
    '''
    df = pd.read_csv(raw_path)

    # Merge category titles if the category_id feature and the category id file path are present
    if cat_id_path and 'category_id' in df.columns: 
        try:    
            with open(cat_id_path, 'r') as cat_id:
                category_data = json.load(cat_id)
            df = merge_with_categories(category_data, df)
        except FileNotFoundError:
            print("Category Json file not found: skipping category merge step ")
    # Drops features with no predictive power or contain post publishing information if present 
    columns_to_drop = ['video_id', 'category_id', 'thumbnail_link', 'channel_title', 'video_error_or_removed' ,
              'comment_count', 'likes', 'dislikes' ,'trending_date']
    df = df.drop([to_drop for to_drop in columns_to_drop if to_drop in df.columns], axis = 1 )

    # Fills NAs of description with empty strings 
    if 'description' in df.columns:
        df['description'] = df['description'].fillna('')


    # Feature Engineering
    if 'description' in df.columns:
        df['description_length'] = df['description'].str.len()

    if 'title' in df.columns:
        df['title_length'] = df['title'].str.len()

    if 'tags' in df.columns:
        df['tag_count'] = df['tags'].str.split('|').str.len()
    
    if 'publish_time' in df.columns:
        df['publish_time'] = pd.to_datetime(df['publish_time'])
        df['week_day'] = df['publish_time'].dt.day_name()

        df['hour'] = df['publish_time'].dt.hour
    
    # Dropping unusable features used in feature engineering 
    df = df.drop(['publish_time', 'description', 'tags', 'title'], axis = 1)

    # Encoding the categorical feature as one hot dummy variables 
    df = pd.get_dummies(data = df, columns = [ 'week_day', 'hour', 'category_title' ], drop_first = True  )

    # Log transforming the target variable to a count for skew  
    if 'views' in df.columns:
        df['log_views'] = np.log1p(df['views'])
        df = df.drop(['views'], axis = 1 )

    # Train/test split 
    train_df, test_df = train_test_split(df, test_size= 0.2 , random_state= 42)

    # Scaling features to prepare for modeling. Scaling without the mean to preserve sparsity
    scalar = StandardScaler(with_mean= False)
    scalar.fit(train_df)
    train_df = pd.DataFrame(scalar.transform(train_df), columns = df.columns)
    test_df = pd.DataFrame(scalar.transform(test_df), columns = df.columns)

    # Export processed CSV to processed file 
    train_df.to_csv(f'{processed_path}/training_data.csv', index = False, encoding= 'utf-8')
    test_df.to_csv(f'{processed_path}/test_data.csv', index = False, encoding= 'utf-8')
