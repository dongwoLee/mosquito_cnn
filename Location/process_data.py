import csv
import calendar
import datetime
import pandas as pd

standardDate=datetime.date(2011,6,1)

dateList = []


for i in range(0,31):
    end_date = standardDate+datetime.timedelta(days=-i)
    (dateList.append(str(end_date)))

matrix = []
with open('Salesio_change_1.csv','r') as f:
    reader = csv.reader(f,delimiter=',')
    for row in reader:
        matrix.append(row)

data = pd.read_csv('Salesio_change_1.csv')














