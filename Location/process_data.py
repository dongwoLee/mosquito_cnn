import csv
import calendar
import datetime

date_1=datetime.date(2011,5,2)


for i in range(1,31):
    end_date = date_1+datetime.timedelta(days=i)
    print (end_date)
