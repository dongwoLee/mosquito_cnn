import csv
import calendar
import datetime
import pandas as pd

date_1=datetime.date(2011,6,1)

dateList = []

for i in range(0,31):
    end_date = date_1+datetime.timedelta(days=-i)
    (dateList.append(end_date))
    print(end_date)



data = pd.read_csv('Salesio_change_1.csv')




