#scrapes earning call data from Seeking Alpha

import requests
import time
from bs4 import BeautifulSoup
doc_id=0
def get_date(c):
    end = c.find('|')
    return c[0:end-1]

def get_ticker(c):
    beg = c.find('(')
    end = c.find(')')
    return c[beg+1:end]

def grab_page(url):
    print("attempting to grab page: " + url)
    page = requests.get(url)
    page_html = page.text
    soup = BeautifulSoup(page_html)
    try:
        heading = soup.find('article').find('header').find('h1')
    except:
        heading = None
    content = soup.find(id="a-body")

    if (heading == None) or (type(content) == "NoneType"):
        print("skipping this link, no content here, or the site may have blocked us from accessing the data")
        return
    else:
        text = content.text
        htext = heading.text.replace(" ", "")
        filename = str(docid) + "" + htext
        doc_id += 1
        file = open(filename.lower() + ".txt", 'w')
        file.write(text)
        file.close
        print(filename.lower()+ " sucessfully saved")

def process_list_page(i):
    origin_page = "https://seekingalpha.com/earnings/earnings-call-transcripts" + "/" + str(i)
    print("getting page " + origin_page)
    time.sleep(1)
    page = requests.get(origin_page)
    page_html = page.text
    soup = BeautifulSoup(page_html)
    alist = soup.find_all("li",{'class':'list-group-item article'})
    for i in range(0,len(alist)):
        url_ending = alist[i].find("h3").find("a").attrs['href']
        url = "https://seekingalpha.com/" + url_ending
        grab_page(url)
        time.sleep(.5)

for i in range(1,10000): #choose what pages of earnings to scrape
    process_list_page(i)