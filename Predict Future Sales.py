#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv(r'C:\Users\燕子\Desktop\competitive-data-science-predict-future-sales\sales_train.csv')
test= pd.read_csv(r'C:\Users\燕子\Desktop\competitive-data-science-predict-future-sales\test.csv')
shops = pd.read_csv(r'C:\Users\燕子\Desktop\competitive-data-science-predict-future-sales\shops.csv')
items = pd.read_csv(r'C:\Users\燕子\Desktop\competitive-data-science-predict-future-sales\items.csv')
cat = pd.read_csv(r'C:\Users\燕子\Desktop\competitive-data-science-predict-future-sales\item_categories.csv')
train.head()


# In[2]:


test.head()


# In[8]:


#查找测试集里面出现了训练集没有的商品
test[~test['shop_id'].isin(train['shop_id'].unique())]
test[~test['item_id'].isin(train['item_id'].unique())]['item_id'].unique()[:10]


# In[9]:


shops.head()


# In[10]:


# 查看商店名称相同但ID不同的店铺
test[test['shop_id'].isin([39, 40, 10, 11, 0, 57, 58, 1, 12 ,56])]['shop_id'].unique()


# In[11]:


#测试集中没有包含同一商店的不同ID， 需要对训练集重复商店的不同ID进行修改
shop_id_map = {11: 10, 0: 57, 1: 58, 40: 39}
train.loc[train['shop_id'].isin(shop_id_map), 'shop_id'] = train.loc[train['shop_id'].isin(shop_id_map), 'shop_id'].map(shop_id_map)
train.loc[train['shop_id'].isin(shop_id_map), 'shop_id']


# In[12]:


train.loc[train['shop_id'].isin([39, 40, 10, 11, 0, 57, 58, 1]), 'shop_id'].unique()


# In[13]:


shops['shop_city'] = shops['shop_name'].map(lambda x:x.split(' ')[0].strip('!'))
shop_types = ['ТЦ', 'ТРК', 'ТРЦ', 'ТК', 'МТРЦ']
shops['shop_type'] = shops['shop_name'].map(lambda x:x.split(' ')[1] if x.split(' ')[1] in shop_types else 'Others')
shops.loc[shops['shop_id'].isin([12, 56]), ['shop_city', 'shop_type']] = 'Online'
shops.head(13)


# In[14]:


# 对商店信息进行编码，降低模型训练的内存消耗
shop_city_map = dict([(v,k) for k, v in enumerate(shops['shop_city'].unique())])
shop_type_map = dict([(v,k) for k, v in enumerate(shops['shop_type'].unique())])
shops['shop_city_code'] = shops['shop_city'].map(shop_city_map)
shops['shop_type_code'] = shops['shop_type'].map(shop_type_map)
shops.head(7)


# In[16]:


# 分析有没有重复名称不同ID的商品
items['item_name'] = items['item_name'].map(lambda x: ''.join(x.split(' ')))
duplicated_item_name = items[items['item_name'].duplicated()]
duplicated_item_name 


# In[17]:


duplicated_item_name_rec = items[items['item_name'].isin(duplicated_item_name['item_name'])]  # 6个商品相同名字不同id的记录
duplicated_item_name_rec


# In[18]:


#查看测试集里面包含了哪些重复项
test[test['item_id'].isin(duplicated_item_name_rec['item_id'])]['item_id'].unique()


# In[19]:


#由于测试集包含了2个同名不同id的商品，需要把训练集里小的ID值都映射为对应较大的ID值
old_id = duplicated_item_name_rec['item_id'].values[::2]
new_id = duplicated_item_name_rec['item_id'].values[1::2]
old_new_map = dict(zip(old_id, new_id))
old_new_map


# In[20]:


train.loc[train['item_id'].isin(old_id), 'item_id'] = train.loc[train['item_id'].isin(old_id), 'item_id'].map(old_new_map)
train[train['item_id'].isin(old_id)]


# In[21]:


# 旧id成功替换成新id
train[train['item_id'].isin(duplicated_item_name_rec['item_id'].values)]['item_id'].unique()  


# In[22]:


# 检查同一个商品是否分了不同类目
items.groupby('item_id').size()[items.groupby('item_id').size() > 1]


# In[24]:


#查看商品类目是否有重复项
cat[cat['item_category_name'].duplicated()]


# In[25]:


#拆分商品类目大类
cat['item_type'] = cat['item_category_name'].map(lambda x: 'Игры' if x.find('Игры ')>0 else x.split(' -')[0].strip('\"')) 
cat.iloc[[32, 33, 34, -3, -2, -1]] 


# In[26]:


cat.iloc[[32,-3, -2], -1] = ['Карты оплаты', 'Чистые носители', 'Чистые носители' ]
cat.iloc[[32,-3, -2]]


# In[27]:


item_type_map = dict([(v,k) for k, v in enumerate(cat['item_type'].unique())])
cat['item_type_code'] = cat['item_type'].map(item_type_map)
cat.head()


# In[28]:


#拆分商品类目小类
cat['sub_type'] = cat['item_category_name'].map(lambda x: x.split('-',1)[-1]) 
cat


# In[30]:


cat['sub_type'].unique()
sub_type_map = dict([(v,k) for k, v in enumerate(cat['sub_type'].unique())])
cat['sub_type_code'] = cat['sub_type'].map(sub_type_map)
cat.head()


# In[31]:


#合并商品和类目数据集
items = items.merge(cat[['item_category_id', 'item_type_code', 'sub_type_code']], on='item_category_id', how='left')
items.head()


# In[32]:


import gc
del cat
gc.collect()


# In[33]:


#观察商品价格和单日销量的分布情况
import seaborn as sns
sns.set()
sns.jointplot('item_cnt_day', 'item_price', train, kind='scatter')


# In[34]:


#过滤明显的离群值
train_filtered = train[(train['item_cnt_day'] < 800) & (train['item_price'] < 70000)].copy()
sns.jointplot('item_cnt_day', 'item_price', train_filtered, kind='scatter')


# In[35]:


#查看价格和销量的异常情况
outer = train[(train['item_cnt_day'] > 400) | (train['item_price'] > 40000)]
outer


# In[36]:


#检查是否需要修改过滤的阈值
outer_set = train_filtered[train_filtered['item_id'].isin(outer['item_id'].unique())].groupby('item_id')
 
fig, ax = plt.subplots(1,1,figsize=(10, 10))
colors = sns.color_palette() + sns.color_palette('bright')
i = 1
for name, group in outer_set:
    ax.plot(group['item_cnt_day'], group['item_price'], marker='o', linestyle='', ms=12, label=name, c=colors[i])
    i += 1
ax.legend()

plt.show()


# In[37]:


train[train['item_id'].isin([13403,7238, 14173])]


# In[38]:


train.loc[train['item_id']==13403].boxplot(['item_cnt_day', 'item_price'])


# In[39]:


#查看400到520中间的商品的销量情况
m_400 = train[(train['item_cnt_day'] > 400) & (train['item_cnt_day'] < 520)]['item_id'].unique()
n = m_400.size
fig, axes = plt.subplots(1,n,figsize=(n*4, 6))
for i in range(n):
    train[train['item_id'] == m_400[i]].boxplot(['item_cnt_day'], ax=axes[i])
    axes[i].set_title('Item%d' % m_400[i])
plt.show()


# In[40]:


filtered = train[(train['item_cnt_day'] < 400) & (train['item_price'] < 45000)].copy()
filtered.head()


# In[42]:


filtered.drop(index=filtered[filtered['item_id'].isin([7238, 14173])].index, inplace=True)
del train, train_filtered
gc.collect()


# In[43]:


#查看有没有小于0的id或者价格
(filtered[['date_block_num', 'shop_id','item_id', 'item_price']] < 0).any()


# In[44]:


# 商品单价小于0的情况
filtered[filtered['item_price'] <= 0]


# In[45]:


filtered.groupby(['date_block_num','shop_id', 'item_id'])['item_price'].mean().loc[4, 32, 2973]


# In[46]:


filtered.loc[filtered['item_price'] <= 0, 'item_price'] = 1249.0  # 用了同一个月同一个商店该商品的均价
filtered[filtered['item_price'] <= 0]  # 检查是否替换成功


# In[47]:


# 下面也给出替换的函数
def clean_by_mean(df, keys, col):
    """
    用同一月份的均值替换小于等于0的值
    keys 分组键；col 需要替换的字段
    """
    group = df[df[col] <= 0]
    mean_price = df.groupby(keys)[col].mean()
    for i, row in group.iterrows:
        record = group.loc[i]
        df.loc[i,col] = mean_price.loc[record[keys[0]], record[keys[1]], record[keys[2]]]
    return df


# In[48]:


# 添加日营业额
filtered['turnover_day'] = filtered['item_price'] * filtered['item_cnt_day']
filtered


# In[49]:


item_sales_monthly = filtered.pivot_table(columns='item_id',
                                          index='date_block_num', 
                                          values='item_cnt_day',
                                          fill_value=0,
                                          aggfunc=sum)
item_sales_monthly.head()


# In[50]:


fig, axes = plt.subplots(1,2, figsize=(20, 8))
item_sales_monthly.sum(1).plot(ax=axes[0], title='Total sales of each month', xticks=[i for i in range(0,34,2)])  # 每月总销量
item_sales_monthly.sum(0).plot(ax=axes[1], title='Total sales of each item')  # 每个商品的总销量
plt.subplots_adjust(wspace=0.2)


# In[51]:


top_sales = item_sales_monthly.sum().sort_values(ascending=False)
top_sales


# In[52]:


test[test['item_id'].isin(top_sales[top_sales<=0].index)]


# In[53]:


top_sales.iloc[0] / item_sales_monthly.sum().sum() * 100  # 销量占比


# In[54]:


item_sales_monthly[top_sales.index[0]].plot(kind='bar', figsize=(12,6))  # 每月销量


# In[55]:


item_turnover_monthly = filtered.pivot_table(index= 'date_block_num',
                                               columns= 'item_id',
                                               values='turnover_day',
                                               fill_value=0,
                                               aggfunc=sum)
item_turnover_monthly.head()


# In[57]:


item_sales_monthly = item_sales_monthly.drop(columns=top_sales[top_sales<=0].index, axis=1)  # 去掉销量为0和负值的商品
item_turnover_monthly = item_turnover_monthly.drop(columns=top_sales[top_sales<=0].index, axis=1)
total_turnover = item_turnover_monthly.sum().sum()
item_turnover_monthly[top_sales.index[0]].sum() / total_turnover * 100


# In[58]:


items[items['item_id']==20949 ]


# In[59]:


(item_sales_monthly > 0).sum(1).plot(figsize=(12, 6))


# In[60]:


# 商品月总销量 / 当月在售商品数量 = 当月在售商品平均销量
item_sales_monthly.sum(1).div((item_sales_monthly > 0).sum(1)).plot(figsize=(12, 6))


# In[61]:


fig, axes = plt.subplots(1,2, figsize=(20, 8))
item_turnover_monthly.sum(1).plot(ax=axes[0], title='Total turnovers of each month', xticks=[i for i in range(0,34,2)])  # 每月总营收
item_turnover_monthly.sum(0).plot(ax=axes[1], title='Total turnovers of each item')  # 每个商品的总营收
plt.subplots_adjust(wspace=0.2)


# In[62]:


top_turnover = item_turnover_monthly.sum().sort_values(ascending=False)
top_turnover


# In[63]:


item_turnover_monthly[top_turnover.index[0]].sum() / total_turnover * 100


# In[64]:


item_sales_monthly[top_turnover.index[0]].sum() / item_sales_monthly.sum().sum() * 100


# In[65]:


item_turnover_monthly[top_turnover.index[0]].plot(kind='bar', figsize=(12, 6))


# In[66]:


item_turnover_monthly[top_turnover.index[0]].div(item_turnover_monthly.sum(1)).plot(figsize=(12, 6),xticks=[i for i in range(0,34,2)])


# In[67]:


items[items['item_id']==top_turnover.index[0]]


# In[68]:


turnover_monthly = item_turnover_monthly.sum(1)
sales_monthly = item_sales_monthly.sum(1)
fig, axe1 = plt.subplots(1, 1, figsize=(16, 6))
axe2 = axe1.twinx()
axe1.plot(turnover_monthly.index, turnover_monthly.values, c='r')

axe2.plot(sales_monthly.index, sales_monthly.values, c='b')
axe2.grid(c='c', alpha=0.3)
axe1.legend(['Monthly Turnover'],fontsize=13, bbox_to_anchor=(0.95, 1))
axe2.legend(['Monthly Sales'],fontsize=13, bbox_to_anchor=(0.93, 0.9))
axe1.set_ylabel('Monthly Turnover', c='r')
axe2.set_ylabel('Monthly Sales', c='b')
plt.show()


# In[69]:


sales_growth = item_sales_monthly.loc[23].sum() - item_sales_monthly.loc[11].sum()
sales_growth_rate = sales_growth / item_sales_monthly.loc[11].sum() * 100
turnover_growth = item_turnover_monthly.loc[23].sum() - item_turnover_monthly.loc[11].sum()
turnover_growth_rate = turnover_growth / item_turnover_monthly.loc[11].sum() * 100
print(
    ' 销售同比增长量为： %.2f ，同比增长率为： %.2f%%;\n' % (sales_growth, sales_growth_rate),
    '营收同比增长量为： %.2f ，同比增长率为： %.2f%%。' % (turnover_growth, turnover_growth_rate)
     )


# In[70]:


dec_set = item_turnover_monthly.loc[[11, 23]]
dec_set


# In[73]:


dec_top = dec_set.loc[:,dec_set.sum() > 5000000]
dec_top  # 年底营收最高的商品


# In[74]:


dec_top.iloc[1, 1:].sum() / dec_set.iloc[1].sum() * 100  # 只在第23月出售的商品其营业额占第23个月所有商品营业额的百分比


# In[75]:


# 13405和13443号商品在第23月销量之和与所有商品总销量的百分比
item_sales_monthly.loc[23,dec_top.columns[1:]].sum() / item_sales_monthly.loc[23].sum() * 100


# In[76]:


(dec_set.iloc[1].sum() - dec_set.iloc[0].sum()) / dec_set.iloc[0].sum() * 100  # 同比增长率


# In[77]:


(dec_set.iloc[1].sum() - dec_set.iloc[0].sum()) / dec_set.iloc[1].sum() * 100  # 增长量占总额的百分比


# In[78]:


item_turnover_monthly[dec_top.columns[1:]].plot(figsize=(12, 6))


# In[82]:


filtered.groupby('shop_id')['item_cnt_day'].sum().sort_values().plot(kind='bar', figsize=(12, 6))


# In[83]:


filtered.groupby('shop_id')['turnover_day'].sum().sort_values().plot(kind='bar', figsize=(12, 6))


# In[84]:


shops[shops['shop_id'].isin([31, 25, 54, 28])]


# In[85]:


filtered = filtered.merge(items.iloc[:,1:], on='item_id', how='left')
filtered.head()


# In[86]:


filtered.groupby('item_category_id')['turnover_day'].sum().sort_values().plot(kind='bar',figsize=(16,6), rot=0)


# In[87]:


filtered.groupby('item_type_code')['turnover_day'].sum().sort_values().plot(kind='bar',figsize=(12,6), rot=0)


# In[88]:


filtered.groupby('sub_type_code')['turnover_day'].sum().sort_values().plot(kind='bar',figsize=(12,6), rot=0)


# In[89]:


filtered.groupby('item_category_id')['item_cnt_day'].sum().sort_values().plot(kind='bar',figsize=(16,6), rot=0)


# In[90]:


filtered.groupby('item_type_code')['item_cnt_day'].sum().sort_values().plot(kind='bar',figsize=(12,6), rot=0)


# In[91]:


filtered.groupby('sub_type_code')['item_cnt_day'].sum().sort_values().plot(kind='bar',figsize=(12,6), rot=0)


# In[92]:


filtered = filtered.merge(shops[['shop_id','shop_city_code','shop_type_code']], on='shop_id', how='left')
filtered.head()


# In[93]:


filtered.groupby('shop_city_code')['turnover_day'].sum().plot(kind='bar',figsize=(12,6))


# In[94]:


filtered.groupby('shop_type_code')['turnover_day'].sum().plot(kind='bar',figsize=(12,6))


# In[95]:


filtered.groupby('shop_city_code')['item_cnt_day'].sum().plot(kind='bar',figsize=(12,6))


# In[96]:


filtered.groupby('shop_type_code')['item_cnt_day'].sum().plot(kind='bar',figsize=(12,6))


# In[97]:


shop_sales_monthly = filtered.pivot_table(index='date_block_num',
                                          columns='shop_id',
                                          values='item_cnt_day',
                                          fill_value=0,
                                          aggfunc=sum)
shop_open_month_cnt = (shop_sales_monthly.iloc[-6:] >  0).sum()  # 有销量的记录
shop_open_month_cnt.head()


# In[98]:


# 最后半年经营月数少于6个月的店铺
shop_c_n = shop_sales_monthly[shop_open_month_cnt[shop_open_month_cnt < 6].index]
shop_c_n.tail(12)


# In[99]:


# 最后半年都正常经营的商店
open_shop = shop_sales_monthly[shop_open_month_cnt[shop_open_month_cnt == 6].index]
open_shop.tail(7)


# In[100]:


# 这些商品在最后半年有几个月有销量
item_selling_month_cnt = (item_sales_monthly.iloc[-6:] >  0).sum() 
item_selling_month_cnt.head()


# In[101]:


# 这些商品在最后半年都没有销量
item_zero = item_sales_monthly[item_selling_month_cnt[item_selling_month_cnt == 0].index]
item_zero.tail(12)


# In[102]:


# 最后半年有销量的商品
selling_item = item_sales_monthly[item_selling_month_cnt[item_selling_month_cnt > 0].index]
selling_item.tail(12) 


# In[103]:


#保留最后6个月正常经营的商店和有销量的商品
cl_set = filtered[filtered['shop_id'].isin(open_shop.columns) & filtered['item_id'].isin(selling_item.columns)]
cl_set


# In[104]:


#统计月销量
from itertools import product
import time
ts = time.time()
martix = []
for i in range(34):
    record = cl_set[cl_set['date_block_num'] == i]
    group = product([i],record.shop_id.unique(),record.item_id.unique())
    martix.append(np.array(list(group)))
            
cols = ['date_block_num', 'shop_id', 'item_id']
martix = pd.DataFrame(np.vstack(martix), columns=cols)
martix


# In[105]:


from itertools import product
import time
ts = time.time()
martix = []
for i in range(34):
    record = filtered[filtered['date_block_num'] == i]
    group = product([i],record.shop_id.unique(),record.item_id.unique())
    martix.append(np.array(list(group)))
            
cols = ['date_block_num', 'shop_id', 'item_id']
martix = pd.DataFrame(np.vstack(martix), columns=cols)

martix


# In[106]:


del cl_set
gc.collect()


# In[107]:


group = filtered.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': np.sum})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)
group


# In[108]:


martix = pd.merge(martix, group, on=['date_block_num', 'shop_id', 'item_id'], how='left')
martix.head()


# In[109]:


test['date_block_num'] = 34
test['item_cnt_month'] = 0
martix = pd.concat([martix.fillna(0), test.drop(columns='ID')], sort=False, ignore_index=True, keys=['date_block_num','shop_id','item_id'])
martix


# In[110]:


#融合商店数据集和商品数据集的特征
martix = martix.merge(shops[['shop_id', 'shop_type_code', 'shop_city_code']], on='shop_id', how='left')
martix = martix.merge(items.drop(columns='item_name'), on='item_id', how='left')
martix


# In[111]:


#添加具体的年份和月份
martix['year'] =  martix['date_block_num'].map(lambda x: x // 12 + 2013)
martix['month'] = martix['date_block_num'].map(lambda x: x % 12)
martix.head()


# In[112]:


# 商品 月销量均值
group = martix.groupby(['date_block_num','item_id']).agg({'item_cnt_month':'mean'})
group.columns = ['item_cnt_month_avg']
group.reset_index(inplace=True)
martix = martix.merge(group, on=['date_block_num', 'item_id'], how='left')
martix.head()


# In[113]:


# 商店 月销量均值
group = martix.groupby(['date_block_num','shop_id']).agg({'item_cnt_month':'mean'})
group.columns = ['shop_cnt_month_avg']
group.reset_index(inplace=True)
martix = martix.merge(group, on=['date_block_num', 'shop_id'], how='left')
martix.head()


# In[114]:


# 类别 月销量均值
group = martix.groupby(['date_block_num','item_category_id']).agg({'item_cnt_month':'mean'})
group.columns = ['cat_cnt_month_avg']
group.reset_index(inplace=True)
martix = martix.merge(group, on=['date_block_num', 'item_category_id'], how='left')
martix.head()


# In[115]:


# 商店-类别 月销量均值
group = martix.groupby(['date_block_num','shop_id','item_category_id']).agg({'item_cnt_month':'mean'})
group.columns = ['shop_cat_cnt_month_avg']
group.reset_index(inplace=True)
martix = martix.merge(group, on=['date_block_num','shop_id','item_category_id'], how='left')
martix.head()


# In[116]:


# 大类 月销量均值
group = martix.groupby(['date_block_num', 'item_type_code']).agg({'item_cnt_month':'mean'})
group.columns = ['itemtype_cnt_month_avg']
group.reset_index(inplace=True)
martix = martix.merge(group, on=['date_block_num', 'item_type_code'], how='left')
martix.head()


# In[117]:


# 小类 月销量均值
group = martix.groupby(['date_block_num', 'sub_type_code']).agg({'item_cnt_month':'mean'})
group.columns = ['subtype_cnt_month_avg']
group.reset_index(inplace=True)
martix = martix.merge(group, on=['date_block_num','sub_type_code'], how='left')
martix.head()


# In[118]:


# 城市-商品 月销量均值
group = martix.groupby(['date_block_num','shop_city_code','item_id']).agg({'item_cnt_month':'mean'})
group.columns = ['city_item_cnt_month_avg']
group.reset_index(inplace=True)
martix = martix.merge(group, on=['date_block_num','shop_city_code','item_id'], how='left')
martix.head()


# In[119]:


# 商店类型-商品 月销量均值
group = martix.groupby(['date_block_num','shop_type_code','item_id']).agg({'item_cnt_month':'mean'})
group.columns = ['shoptype_item_cnt_month_avg']
group.reset_index(inplace=True)
martix = martix.merge(group, on=['date_block_num','shop_type_code','item_id'], how='left')
martix.head()


# In[120]:


del group
gc.collect()


# In[121]:


# 添加销量特征的历史特征
def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df
martix = lag_feature(martix, [1,2,3,6,12], 'item_cnt_month')
martix.head()


# In[122]:


martix = lag_feature(martix, [1,2,3,6,12], 'item_cnt_month_avg')
martix = lag_feature(martix, [1,2,3,6,12], 'shop_cnt_month_avg')
martix.head()


# In[123]:


martix.drop(columns=[ 'item_cnt_month_avg', 'shop_cnt_month_avg'], inplace=True)  # 只保留特征的历史信息
gc.collect()


# In[124]:


martix = lag_feature(martix, [1,2,3,6,12], 'cat_cnt_month_avg')
martix = lag_feature(martix, [1,2,3,6,12], 'shop_cat_cnt_month_avg')
martix.head()


# In[126]:


martix.drop(columns=['cat_cnt_month_avg', 'shop_cat_cnt_month_avg'], inplace=True)
martix = lag_feature(martix, [1,2,3,6,12], 'itemtype_cnt_month_avg')
martix = lag_feature(martix, [1,2,3,6,12], 'subtype_cnt_month_avg')
martix.head()


# In[127]:


martix.drop(columns=['itemtype_cnt_month_avg', 'subtype_cnt_month_avg'], inplace=True)
martix = lag_feature(martix, [1,2,3,6,12], 'city_item_cnt_month_avg')
martix = lag_feature(martix, [1,2,3,6,12], 'shoptype_item_cnt_month_avg')
martix.head()


# In[128]:


martix.drop(columns=[ 'city_item_cnt_month_avg','shoptype_item_cnt_month_avg'], inplace=True)
martix


# In[129]:


martix[martix.columns[:20]].isna().any()


# In[130]:


train_set = martix[martix['date_block_num'] > 11].fillna(0)
train_set


# In[131]:


for col in train_set.columns:
    if col.find('code') >= 0:
        train_set[col] = train_set[col].astype(np.int8)
    elif train_set[col].dtype == 'float64':
        train_set[col] = train_set[col].astype(np.float32)
    elif train_set[col].dtype == 'int64':
        train_set[col] = train_set[col].astype(np.int16)
        
train_set['item_type_code'] = train_set['item_type_code'].astype('category')
train_set['sub_type_code'] = train_set['sub_type_code'].astype('category')
train_set.info()


# In[134]:


#使用lightgbm模型进行训练
import lightgbm as lgb
X_train = train_set[train_set['date_block_num'] < 33].drop(columns=['item_cnt_month'])  # 训练集的样本特征
Y_train = train_set[train_set['date_block_num'] < 33]['item_cnt_month']  # 训练集的样本标签

X_validate = train_set[train_set['date_block_num'] == 33].drop(columns=['item_cnt_month'])  # 校对集
Y_validate = train_set[train_set['date_block_num'] == 33]['item_cnt_month']

X_test = train_set[train_set['date_block_num'] == 34].drop(columns=['item_cnt_month'])  # 测试集
del train_set
gc.collect()


# In[135]:


# 把数据加载为模型适合的数据格式
train_data = lgb.Dataset(data=X_train, label=Y_train)
validate_data = lgb.Dataset(data=X_validate, label=Y_validate)


# In[136]:


# 设置模型训练参数
import time
ts = time.time()
params = {"objective" : "regression", "metric" : "rmse", 'n_estimators':10000, 'early_stopping_rounds':50,
              "num_leaves" : 200, "learning_rate" : 0.01, "bagging_fraction" : 0.9,
              "feature_fraction" : 0.3, "bagging_seed" : 0}
print('Start....', ts)
lgb_model = lgb.train(params, train_data, valid_sets=[train_data, validate_data], verbose_eval=1000) 
print('End...', time.time() - ts)


# In[137]:


# 特征重要性画图
lgb.plot_importance(lgb_model, max_num_features=40, figsize=(12, 8))
plt.title("Featurertances")
plt.show()


# In[138]:


lgb_model.save_model(r'C:\Users\燕子\Desktop\competitive-data-science-predict-future-sales\model_bestscore02.txt')  # 保存模型


# In[139]:


# 根据项目要求，把数据“裁剪”到[0,20]的区间。
Y_test = lgb_model.predict(X_test).clip(0, 20)
Y_test


# In[140]:


X_test['item_cnt_month'] = Y_test
X_test


# In[141]:


#将预测结果合并到测试集
result = pd.merge(test[['ID', 'shop_id', 'item_id']],X_test[['shop_id','item_id','item_cnt_month']], on=['shop_id', 'item_id'], how='left')
result


# In[142]:


result.isna().any()


# In[143]:


result[result.shop_id.isin(shop_c_n.columns)]['shop_id'].unique()


# In[144]:


result.loc[result.item_id.isin(item_zero), 'item_cnt_month'] = 0
result.loc[result.item_id.isin(item_zero), 'item_cnt_month']


# In[145]:


result[['ID','item_cnt_month']].to_csv(r'C:\Users\燕子\Desktop\competitive-data-science-predict-future-sales\submission.csv',sep=',',index=False)


# In[ ]:




