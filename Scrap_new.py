from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from urllib.parse import quote
import html5lib
import time

root = "https://www.google.com/"

def news(link,querys):
    querys = quote(querys)
    req = Request(link, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    soup = BeautifulSoup(webpage, 'html5lib')
    # print(soup)
    for item in soup.find_all('div', attrs={'class': 'ZINbbc luh4tb xpd O9g5cc uUPGi'}):
        title = (item.find('div', attrs={'class': 'BNeawe vvjwJb AP7Wnd'}).get_text())
        title = title.replace(",", "")
        print(title)
        document = open("csv/Data.csv", "a")
        document.write("{} \n".format(title))
        document.close()

    next = soup.find(attrs={'aria-label':'Next page'})
    next = (next['href'])
    link = root + next
    time.sleep(3)
    news(link,querys)



if __name__ == '__main__':
    querys = "blockchain"
    link = f"https://www.google.com/search?q={querys}&hl=en&tbas=0&tbm=nws&source=lnt&tbs=sbd:1&sa=X&ved=2ahUKEwiApO-EmfH1AhXD6mEKHXuLA6YQpwV6BAgBECE&biw=962&bih=967&dpr=1"
    news(link,querys)


