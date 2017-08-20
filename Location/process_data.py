import csv
import calendar
import datetime
import pandas as pd

standardDate=datetime.date(2011,6,1)
lastDate = datetime.date(2011,10,31)

daterange = pd.date_range(standardDate,lastDate)

dateList = []

for single_date in daterange:
    dateList.append(str(single_date))
    # for i in range(0,31):
    #     end_date = single_date+datetime.timedelta(days=-i)
    #     (dateList.append(str(end_date)))

for i in range(len(dateList)):
    dateList[i]=dateList[i][0:10]

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
            result[-1]=mos ## change mosquito data
            with open("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Salesio_"+str(dateList[i])+".csv","a") as f:
                wr = csv.writer(f)
                wr.writerow(result)
















