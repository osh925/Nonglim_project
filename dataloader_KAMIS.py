import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

### DATA LOADING ###



# print(price22) // 22년도 데이터는 훼손이 심각하여 일단 배제. 나중에 노가다로 끼워맞추던가 아니면 자료 없다고 구라치기 ㄱ

####################
seoul_apple = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/KAMIS_dataset/seoul/seoul_apple.csv', encoding='UTF-8')
seoul_baechu = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/KAMIS_dataset/seoul/seoul_baechu.csv', encoding='UTF-8')
seoul_fish = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/KAMIS_dataset/seoul/seoul_fish.csv', encoding='UTF-8')
seoul_pig = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/KAMIS_dataset/seoul/seoul_pig.csv', encoding='UTF-8')
seoul_rice = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/KAMIS_dataset/seoul/seoul_rice.csv', encoding='UTF-8')

busan_apple = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/KAMIS_dataset/busan/busan_apple.csv', encoding='UTF-8')
busan_baechu = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/KAMIS_dataset/busan/busan_baechu.csv', encoding='UTF-8')
busan_fish = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/KAMIS_dataset/busan/busan_fish.csv', encoding='UTF-8')
busan_pig = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/KAMIS_dataset/busan/busan_pig.csv', encoding='UTF-8')
busan_rice = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/KAMIS_dataset/busan/busan_rice.csv', encoding='UTF-8')

daegu_apple = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/KAMIS_dataset/daegu/daegu_apple.csv', encoding='UTF-8')
daegu_baechu = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/KAMIS_dataset/daegu/daegu_baechu.csv', encoding='UTF-8')
daegu_fish = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/KAMIS_dataset/daegu/daegu_fish.csv', encoding='UTF-8')
daegu_pig = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/KAMIS_dataset/daegu/daegu_pig.csv', encoding='UTF-8')
daegu_rice = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/KAMIS_dataset/daegu/daegu_rice.csv', encoding='UTF-8')

######################

# plt.figure(figsize=(15, 10))
# plt.plot(DACON_TRAIN['date'], DACON_GARLIC, linestyle='-')

# # plt.xticks(rotation=45)
# # plt.grid(True)
# # plt.ylim(0, max(gwangjang_apple_sum['가격(원)']) * 1.5) 
# plt.show()

# print(gwangjang_apple_df)
# df0.rename(columns={'Date':'Price'})
# ts0 = df0.copy()
# ts0['Date'] = pd.to_datetime(ts0.Date, format='%Y-%m-%d')
# ts0.info()