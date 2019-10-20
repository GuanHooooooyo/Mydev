import scrapy
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas


'''使用pandas抓取匯率牌價資訊'''
df = pandas.read_html('https://rate.bot.com.tw/xrt/all/day')
currency = df[0]
#print(currency)
print(type(currency))
currency = currency.ix[:,0:5]
currency.columns = [u'幣別',u'現金匯率-本行買入',u'現金匯率-本行賣出',u'即期匯率-本行買入',u'即期匯率-本行賣出']
currency[u'幣別'] = currency[u'幣別'].str.extract('\((\w+)\)')
print(currency)
currency.to_excel('currency.xlsx')

'''使用selenium開啟網頁並抓取資訊'''
chromepath = 'D:/test lab/chromedriver.exe'
driver = webdriver.Chrome(chromepath)
driver.get('https://www.google.com')
q = driver.find_element_by_name('q')
q.send_keys('高雄科技大學')

q.send_keys(Keys.RETURN)
bf = BeautifulSoup(driver.page_source,'lxml')

# for ele in bf.select('div#rso a h3'):
#     print(ele.text)

import time
for page in range(3):
    driver.find_element_by_link_text('下一頁').click()
    soup = BeautifulSoup(driver.page_source, 'lxml')
    for ele in bf.select('div#rso a h3'):
        print(ele.text)
    time.sleep(1)


