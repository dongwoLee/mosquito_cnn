import urllib.request
from bs4 import BeautifulSoup
from selenium import webdriver
import csv

URL = 'http://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=100&sid2=266&oid=023&aid=0003303155'
html = urllib.request.urlopen(URL)
def get_text(URL):
    soup = BeautifulSoup(html, 'html.parser', from_encoding='utf-8')

    write = soup.select('.u_cbox_info')
    print (write)
    time = soup.findAll('ul',{'class':'u_cbox_list'})
    print(time)
    text = ''

    # for tag in nick:
    #     text = text + str(tag.a['u_cbox_nick'])+'\t '
    #
    # return text

def main():
	# open_output_file = open('output01.csv', 'w')
	# result_text = get_text(URL)
	# open_output_file.write(result_text)
	# open_output_file.close()
    get_text(URL)
if __name__ == '__main__':
    main()