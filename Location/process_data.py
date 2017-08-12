import csv
import calendar
import datetime
import pandas as pd

date_1=datetime.date(2011,5,2)

dateList = []

for i in range(1,31):
    end_date = date_1+datetime.timedelta(days=i)
    print(end_date)


#data = pd.read_csv('Salesio_change_1.csv')


