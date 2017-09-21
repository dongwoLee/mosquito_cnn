import pandas as pd
import numpy as np
from itertools import islice

def chunk(it,size):
	it = iter(it)
	return iter(lambda:tuple(islice(it,size)),())

def split_30Days(file):
	data = pd.read_csv(file) #wholeData 
	columnList = data.columns.values.tolist()
	columnList = columnList[1:len(columnList)]
	
	wholeColumnList = []
	for i in range(len(columnList)):
		wholeColumnList.append(columnList[i:i+30])

	for j in range(len(wholeColumnList)):
		if(len(wholeColumnList[j]) == 30):
			result = []
			for k in range(len(wholeColumnList[j])):
				temp_data = pd.DataFrame(data[wholeColumnList[j][k]])
				result.append(temp_data)
			result = pd.concat(result, axis=1)
			result.to_csv('C://Users/dw/Desktop/mosquito_cnn/Location_allDate/split_csv/'+str(wholeColumnList[j][0])+'.csv')

if __name__ == "__main__":
	split_30Days('C:/Users/dw/Desktop/mosquito_cnn/meterology/factor_data_result.csv')
