import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

### DATA LOADING ###

price23_24 = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/seoul/price_23_24.csv', encoding='ANSI')
# price22 = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/seoul/price_22.csv', encoding='ANSI')
price21 = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/seoul/price_21.csv', encoding='ANSI')
price20 = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/seoul/price_20.csv', encoding='ANSI')
price19 = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/seoul/price_19.csv', encoding='ANSI')
price18 = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/seoul/price_18.csv', encoding='ANSI')
price13_17 = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/seoul/price_13_17.csv', encoding='ANSI')

DACON_TRAIN = pd.read_csv('./24_2R_DATASCIENCE_PROJECT/data/train.csv', encoding='ANSI')

# print(price22) // 22년도 데이터는 훼손이 심각하여 일단 배제. 나중에 노가다로 끼워맞추던가 아니면 자료 없다고 구라치기 ㄱ

####################


### PREPROCESSING ### : 광장시장 사과 예시
gwangjang13_17 = price13_17[price13_17['시장/마트 이름'] == '광장시장']
gwangjang_apple_13_17 = gwangjang13_17[gwangjang13_17['품목 번호'] == 305] # 사과 1개

gwangjang18 = price18[price18['시장/마트 이름'] == '광장시장']
gwangjang_apple_18 = gwangjang18[gwangjang18['품목 번호'] == 305] # 사과 1개

gwangjang19 = price19[price19['시장/마트 이름'] == '광장시장']
gwangjang_apple_19 = gwangjang19[gwangjang19['품목 번호'] == 305] # 사과 1개

gwangjang20 = price20[price20['시장/마트 이름'] == '광장시장']
gwangjang_apple_20 = gwangjang20[gwangjang20['품목 번호'] == 305] # 사과 1개

gwangjang21 = price21[price21['시장/마트 이름'] == '광장시장']
gwangjang_apple_21 = gwangjang21[gwangjang21['품목 번호'] == 305] # 사과 1개

gwangjang23_24 = price23_24[price23_24['시장/마트 이름'] == '광장시장']
gwangjang_apple_23_24 = gwangjang23_24[gwangjang23_24['품목 번호'] == 2] # 사과 1개

DACON_GARLIC = DACON_TRAIN['마늘_가격(원/kg)']

gwangjang_apple_sum = [gwangjang_apple_23_24, gwangjang_apple_21, gwangjang_apple_20, gwangjang_apple_19, 
                       gwangjang_apple_18, gwangjang_apple_13_17]
gwangjang_apple_sum = pd.concat(gwangjang_apple_sum, ignore_index=True)

gwangjang_apple_sum = gwangjang_apple_sum.iloc[::-1].reset_index(drop=True)

# 공무원 이 씹새끼들 왜 년도마다 품목번호가 달라요????????????

# print(gwangjang_apple_sum)

# plt.figure(figsize=(15, 10))
# plt.plot(DACON_TRAIN['date'], DACON_GARLIC, linestyle='-')

# # plt.xticks(rotation=45)
# # plt.grid(True)
# # plt.ylim(0, max(gwangjang_apple_sum['가격(원)']) * 1.5) 
# plt.show()

gwangjang_apple_df = gwangjang_apple_sum.loc[:, ['점검일자', '가격(원)']]
gwangjang_apple_df['가격(원)'] = gwangjang_apple_df['가격(원)'].where(gwangjang_apple_df['가격(원)'] != 0, 
                                                                    gwangjang_apple_df['가격(원)'].shift(-1))
DACON_GARLIC = DACON_GARLIC.where(DACON_GARLIC != 0, DACON_GARLIC.shift(-1))

# print(gwangjang_apple_df)
# df0.rename(columns={'Date':'Price'})
# ts0 = df0.copy()
# ts0['Date'] = pd.to_datetime(ts0.Date, format='%Y-%m-%d')
# ts0.info()