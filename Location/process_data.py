import csv
import calendar
import datetime
import pandas as pd

standardDate=datetime.date(2011,6,1)

dateList = []


for i in range(0,31):
    end_date = standardDate+datetime.timedelta(days=-i)
    (dateList.append(str(end_date)))

data = pd.read_csv('Salesio_change_1.csv')
mos = (data[str(standardDate)].tolist())
mos = mos[-1] ## standard mosquito data

wholeDate = []
with open('Salesio_change_1.csv','r') as f:
    reader = csv.reader(f,delimiter=',')
    for row in reader:
        wholeDate.append(row)

for i in range(len(dateList)):
    for j in range(len(wholeDate[0])):
        if(dateList[i]==wholeDate[0][j]):
            result=data[str(dateList[i])].tolist()
            result[-1]=mos
            with open("/Users/leedongwoo/Desktop/mosquito_cnn/Location_allDate/Salesio_06_01.csv","a") as f:
                wr = csv.writer(f)
                wr.writerow(result)
















