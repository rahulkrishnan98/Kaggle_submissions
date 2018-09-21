import pandas as pd

my_file=pd.read_csv('sales_train_v2.csv')

df=my_file.drop('date',1)
df=df.drop('item_price',1)

df=df.pivot_table(df,index=["date_block_num","shop_id","item_id"]).reset_index()
df=df.rename(columns={'item_cnt_day':'item_cnt_month'})
print(df.head())

#updated dataset is written to new file
df.to_csv('newdata.csv')
#Note: Delete indexing in resultant file




