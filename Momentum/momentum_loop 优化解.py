
import os
import pandas as pd
import numpy as np
import wrds
import matplotlib.pyplot as plt
from pandas.tseries.offsets import *
from scipy import stats
import xlsxwriter
import datetime as dt


def modify_momr(row):
    if 'winners' in row:
        return 'winners'
    elif 'losers' in row:
        return 'losers'
    elif 'long_short' in row:
        return 'long short'


conn = wrds.Connection()

if __name__ == "__main__":

    crsp_m = conn.raw_sql("""
                      select a.permno, a.permco, b.ncusip, a.date,
                      b.shrcd, b.exchcd, b.siccd,
                      a.ret, a.vol, a.shrout, a.prc, a.cfacpr, a.cfacshr
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date between '01/01/1963' and '12/31/1989'
                      and b.exchcd between -2 and 2
                      and b.shrcd between 10 and 11
                      """)
    _tmp_crsp = crsp_m[['permno', 'date', 'ret']].sort_values(['permno', 'date']) \
        .set_index('date')



    # data_file1 = os.path.join('.', "crsp_m.csv")
    # crsp_m = pd.read_csv(data_file1)

    # data_file2 = os.path.join('.', "_tmp_crsp.csv")
    # _tmp_crsp = pd.read_csv(data_file2)

    ###经典的数据预处理方式工作
    _tmp_crsp = _tmp_crsp.set_index('date')
    _tmp_crsp['ret'] = _tmp_crsp['ret'].fillna(0)

    ###构建 logreturn 方便rolling
    _tmp_crsp['logret'] = np.log(1 + _tmp_crsp['ret'])

    count_first_layer = 1
    for K in [3,6,9,12]:
        print(K)
        count_second_layer = 1
        for J in [3,6,9,12]:
            print(J)
            print(count_second_layer)

            umd = _tmp_crsp.groupby(['permno'])['logret'].rolling(J, min_periods=J).sum()

            umd = umd.reset_index()
            umd['cumret'] = np.exp(umd['logret']) - 1

            umd = umd.dropna(axis=0, subset=['cumret'])

            umd['momr'] = umd.groupby('date')['cumret'].transform(lambda x: pd.qcut(x, 10, labels=False))


            umd.momr = umd.momr.astype(int)
            umd['momr'] = umd['momr'] + 1



            # Corrected previous version month end line up issue
            # First lineup date to month end date medate
            # Then calculate hdate1 and hdate2 using medate
            umd['date'] = pd.to_datetime(umd['date'])

            # 计算持有期的开始和结束日期
            umd['form_date'] = umd['date']
            umd['medate'] = umd['date'] + MonthEnd(0)
            umd['hdate1'] = umd['medate'] + MonthBegin(2)
            umd['hdate2'] = umd['medate'] + MonthEnd(K+1)
            umd = umd[['permno', 'form_date', 'momr', 'hdate1', 'hdate2','medate']]

            tmp1 = umd
            tmp1['hdate1_m'] = pd.DatetimeIndex(tmp1['hdate1']).month
            tmp1['hdate2_m'] = pd.DatetimeIndex(tmp1['hdate2']).month

            tmp1['mgap'] = tmp1.hdate2_m - tmp1.hdate1_m
            tmp1['mgap'] = np.where(tmp1.mgap > 0, tmp1.mgap, tmp1.mgap + 12)

            _tmp_ret = crsp_m[['permno', 'date', 'ret']]

            port = pd.DataFrame()
            n = 100000
            for i in range(0, _tmp_ret.shape[0], n):
                # 每次迭代100000行
                chunks = _tmp_ret.iloc[i:i + n]
                # print(chunks)
                merged = umd.merge(chunks, on=['permno'], how='inner')
                # print(merged)
                merged = merged[(merged['hdate1'] <= merged['date']) & (merged['date'] <= merged['hdate2'])]
                port = pd.concat([port, merged])

            umd2 = port.sort_values(by=['date', 'momr', 'form_date', 'permno']).drop_duplicates()
            umd3 = umd2.groupby(['date', 'momr', 'form_date'])['ret'].mean().reset_index()
            print(umd3.columns)
            # print(umd3)

            umd3['date'] = pd.to_datetime(umd3['date'])

            start_yr = umd3['date'].dt.year.min() + 2
            #获取数据集中的最小年份，并加上2，表示从形成期结束后的第二年 开始计算。
            # 这样做是为了避免在形成期初期的数据不够稳定时使用数据。

            umd3 = umd3[umd3['date'].dt.year >= start_yr]
            umd3 = umd3.sort_values(by=['date', 'momr'])
            print(umd3)


            # Create one return series per MOM group every month

            ewret = umd3.groupby(['date', 'momr'])['ret'].mean().reset_index()
            ewstd = umd3.groupby(['date', 'momr'])['ret'].std().reset_index()
            ewret = ewret.rename(columns={'ret': 'ewret'})
            ewstd = ewstd.rename(columns={'ret': 'ewretstd'})
            ewretdat = pd.merge(ewret, ewstd, on=['date', 'momr'], how='inner')
            ewretdat = ewretdat.sort_values(by=['momr'])

            # Transpose portfolio layout to have columns as portfolio returns
            ewretdat2 = ewretdat.pivot(index='date', columns='momr', values='ewret')
            # print(ewretdat2)

            # Add prefix port in front of each column
            ewretdat2 = ewretdat2.add_prefix('port')
            ewretdat2 = ewretdat2.rename(columns={'port1': f'losers{J}', 'port10': f'winners{J}'})
            ewretdat2[f'long_short{J}'] = ewretdat2[f'winners{J}'] - ewretdat2[f'losers{J}'] ##这是新加了一列！表示差值

            # Compute Long-Short Portfolio Cumulative Returns
            ewretdat3 = ewretdat2
            ewretdat3['1+losers'] = 1 + ewretdat3[f'losers{J}']
            ewretdat3['1+winners'] = 1 + ewretdat3[f'winners{J}']
            ewretdat3['1+ls'] = 1 + ewretdat3[f'long_short{J}']

            ewretdat3['cumret_winners'] = ewretdat3['1+winners'].cumprod() - 1
            ewretdat3['cumret_losers'] = ewretdat3['1+losers'].cumprod() - 1
            ewretdat3['cumret_long_short'] = ewretdat3['1+ls'].cumprod() - 1


            ewretdat3.to_csv(f"./ewretdat3{J}{K}.csv")
            print(f'ewretdat3{J}{K}输出了')

            mom_mean = ewretdat3[[f'winners{J}', f'losers{J}', f'long_short{J}']].mean().to_frame()
            mom_mean = mom_mean.rename(columns={0: 'mean'}).reset_index()

            # T-Value and P-Value
            t_losers = pd.Series(stats.ttest_1samp(ewretdat3[f'losers{J}'], 0.0)).to_frame().T
            t_winners = pd.Series(stats.ttest_1samp(ewretdat3[f'winners{J}'], 0.0)).to_frame().T
            t_long_short = pd.Series(stats.ttest_1samp(ewretdat3[f'long_short{J}'], 0.0)).to_frame().T

            t_losers['momr'] = f'losers{J}'
            t_winners['momr'] = f'winners{J}'
            t_long_short['momr'] = f'long_short{J}'

            t_output = pd.concat([t_winners, t_losers, t_long_short]) \
                .rename(columns={0: 't-stat', 1: 'p-value'})

            # Combine mean, t and p
            mom_output = pd.merge(mom_mean, t_output, on=['momr'], how='inner')

            print(mom_output)


            #######如何生成漂亮表格呢！
            ####### 关于如何将每个动量组合重复两行：一行是均值，另一行是t值）
            ####### mean 和 t-stat 纵向堆叠，形成一列数据
            mom_output = pd.DataFrame({
                'momr': mom_output['momr'].repeat(2).values,
                f'{K}': mom_output[['mean', 't-stat']].stack().values
            })

            '''
            where(mom_output.index % 2 == 0) 表示在偶数行保留原来的 momr 名称（例如，winners、losers）
            ，因为这些行对应均值。
            在奇数行，将 momr 后加上 _t-stat（如 winners_t-stat、losers_t-stat），
            以区分这些行表示的是 t 值。
            '''
            mom_output['momr'] = mom_output['momr'].where(mom_output.index % 2 == 0, mom_output['momr'] + '_t-stat')

            ##将形成期J加入输出
            mom_output['J'] = [J,J,J,J,J,J]

            ## 将 J 列设置为索引，以便后续更容易在不同的形成期之间进行合并
            mom_output = mom_output.set_index('J')
            print(mom_output)

            if K == 3 and count_second_layer == 1:
                k_3 = mom_output
            elif K == 3 and count_second_layer != 1:
                k_3 = pd.concat([k_3,mom_output])

            elif K == 6 and count_second_layer == 1:
                k_6 = mom_output
            elif K == 6 and count_second_layer != 1:
                k_6 = pd.concat([k_6,mom_output])

            elif K == 9 and count_second_layer == 1:
                k_9 = mom_output
            elif K == 9 and count_second_layer != 1:
                k_9 = pd.concat([k_9,mom_output])

            elif K == 12 and count_second_layer == 1:
                k_12 = mom_output
            elif K == 12 and count_second_layer != 1:
                k_12 = pd.concat([k_12,mom_output])

            count_second_layer += 1
        count_first_layer += 1
    print(k_3, k_6, k_9, k_12)
    k_3 = k_3.reset_index()


    k_3_6 = pd.merge(k_3, k_6[['momr', '6']], how='left', on='momr')
    k_3_6_9 = pd.merge(k_3_6, k_9[['momr', '9']], how='left', on='momr')
    k_3_6_9_12 = pd.merge(k_3_6_9, k_12[['momr', '12']], how='left', on='momr')

    print(k_3_6_9_12)
    k_3_6_9_12['momr'] = k_3_6_9_12['momr'].apply(modify_momr)

    ## 1::2 是一个步长为2的切片操作，表示选择奇数行。
    k_3_6_9_12.loc[1::2, 'momr'] = ''
    ## 将这些奇数行的 momr 设置为空白，这可能是为了美化输出格式，
    # 避免 t-stat 的行重复显示组合名称。

    k_3_6_9_12['J'] = k_3['J']
    k_3_6_9_12.loc[1::2, 'J'] = ''

    ##插入K=列：插入一个空列 K=，这是为了表格格式化或美观，
    # 最终输出时显示不同持有期（K）的表现
    k_3_6_9_12.insert(2,'K=','')

    # 无返回值，是就地修改
    print(k_3_6_9_12[['J','momr','K=','3','6','9','12']])

    a = k_3_6_9_12[['J', 'momr', 'K=', '3', '6', '9', '12']]

    a.to_excel('./result.xlsx')