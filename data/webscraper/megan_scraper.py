import requests
import time
import threading
import multiprocessing
from timeout import timeout, TimeoutError
from selenium import webdriver
from bs4 import BeautifulSoup
from pymongo import MongoClient

county_url_format = 'http://www.meganslaw.ca.gov/cgi/prosoma.dll?SelectCounty={}&searchBy=countylist&W6={}&PageNo={}'
start_point = 'http://www.meganslaw.ca.gov/disclaimer.aspx'
profile_url_format = 'http://www.meganslaw.ca.gov/cgi/prosoma.dll?w6={}&searchby=offender&id={}'
with open('counties.txt') as f:
    counties = f.read().split('\n')
# counties = [county for county in counties if len(county.split()) == 1]

DB_NAME = "megan"
COLLECTION_NAME = "so"

client = MongoClient()
db = client[DB_NAME]
coll = db[COLLECTION_NAME]


def get_w6():
    w6 = request_wrapper('http://meganslaw.ca.gov/cgi/prosoma.dll?searchby=curno').content.strip('\r\n')
    return w6

#@timeout(5)
def timed_request(url, queue):
    reqs = queue.get()
    # return requests.get(url)
    r = requests.get(url)
    reqs[url] = r
    queue.put(reqs)


def request_wrapper(url):
    queue = multiprocessing.Queue()
    reqs = {}
    queue.put(reqs)
    while True:
        p = multiprocessing.Process(target=timed_request, args=(url, queue))
        p.start()
        p.join(5)
        if p.is_alive():
            p.terminate()
            p.join()
            continue
        else:
            reqs = queue.get()
            return reqs[url]

    # while True:
    #     try:
    #         return timed_request(url)
    #     except TimeoutError:
    #         pass



def scrape_listings(county, w6, page):
    print 'Starting page {}.'.format(page)
    r = request_wrapper(county_url_format.format(county, w6, page))
    bs = BeautifulSoup(r.content, 'html.parser')
    print 'Got page {}.'.format(page)
    links = bs.findAll('a', text='More Info')
    temp_ids = [link.attrs['href'].split("'")[1] for link in links]
    if len(temp_ids) == 0:
        print 'Something went wrong on page {}. Redoing.'.format(page)
        w6 = re_up_w6()
        w6 = scrape_listings(county, w6, page)
        return w6
    else:
        mongos = [{'id':so} for so in temp_ids]
        coll.insert_many(mongos)
        return w6


def get_ids_by_county(county, w6):
    print 'Finding endpage for {}.'.format(county)
    r = request_wrapper(county_url_format.format(county, w6, 1))
    bs = BeautifulSoup(r.content, 'html.parser')
    try:
        num_offs = int(bs.findAll('br')[1].text.split('of')[1].strip())
        endpage = num_offs / 20 + (1 if num_offs % 20 != 0 else 0)
    except IndexError:
        if test_w6(w6):
            w6 = get_ids_by_county(county, w6)
            return w6
        else:
            w6 = re_up_w6()
            w6 = get_ids_by_county(county, w6)
            return w6
    print 'Endpage found.  There are {} pages.'.format(endpage)
    # ids = []
    print 'Scanning pages for ids.'
    # threads = []
    for page in xrange(1, endpage + 1):
        w6 = scrape_listings(county, w6, page)

    return w6
    #     t = threading.Thread(target=scrape_listings, args=(county, w6, page))
    #     threads.append(t)
    #
    # for thread in threads:
    #     thread.start()
    #
    # for thread in threads:
    #     thread.join()

        # print 'Scanning page {}'.format(page)
        # r = request_wrapper(listing_url_format.format(w6, city, page))
        # bs = BeautifulSoup(r.content, 'html.parser')
        # links = bs.findAll('a', text='More Info')
        # temp_ids = [link.attrs['href'].split("'")[1] for link in links]
        # ids.extend(temp_ids)
        # print 'IDs on this page: \n{}'.format(temp_ids)
        # print 'Total IDs so far: {}'.format(len(ids))
        # if len(temp_ids) == 0:
        #     import pdb; pdb.set_trace()
        #     print 'Something is wrong.'
        #     break
    # return ids

def test_w6(w6):
    r = request_wrapper(county_url_format.format(w6, 'SAN%20FRANCISCO', 1))
    if 'expired' in r.content:
        print 'Bad w6.'
        return False
    else:
        return True

def re_up_w6():
    w6 = get_w6()
    if not test_w6(w6):
        w6 = re_up_w6()
    return w6

def scrape_profile(so_id):
    w6 = re_up_w6()
    print 'Requesting so{}'.format(so_id)
    r = request_wrapper(profile_url_format.format(w6, so_id))
    print 'Got request back.'
    if 'expired' in r.content:
        print 'Refreshing w6'
        w6 = re_up_w6()
        scrape_profile(so_id)
    bs = BeautifulSoup(r.content, 'html.parser')
    fields = bs.findAll('th')
    values = bs.findAll('td')
    adjust = 0
    field_val = {'so_id': so_id}
    for index in xrange(len(fields)):
        if not fields[index].text:
            adjust += 1
        else:
            field_val[fields[index].text] = values[index - adjust].text
    # field_val['Last Known Address:'] = values[-2].text
    redtext = bs.findAll('p', attrs={'class': 'redtextbold'})
    for i, red in enumerate(redtext):
        field_val['redtext{}'.format(i)] = red.text
    coll.insert_one(field_val)
    print 'Mongo updated for so{}'.format(so_id)

if __name__ == '__main__':
    w6 = get_w6()

    all_ids = coll.find({})
    dist_ids = all_ids.distinct('id')
    # num_done = 0
    threads = []
    for so_id in dist_ids:
        t = threading.Thread(target=scrape_profile, args=(so_id,))
        threads.append(t)
        # print 'Done with {}.'.format(num_done)
        # scrape_profile(so_id)
        # num_done += 1

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # for county in counties:
    #     get_ids_by_county(county, w6)

    # ids = coll.find({})
    # dist_ids = ids.distinct('id')
    # num_ids = dist_ids.count()
    # for so_id in dist_ids:
    #     r = request_wrapper(profile_url_format.format(w6, so_id))
    #     bs = BeautifulSoup(r.content, 'html.parser')
