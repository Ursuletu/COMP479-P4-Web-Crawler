from urllib.request import urlopen, Request

from bs4 import BeautifulSoup
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from afinn import Afinn

import urllib.robotparser
import math


# Returns true if user crawler can fetch the URL, false otherwise
def can_crawl(url):
    try:
        rp = urllib.robotparser.RobotFileParser()
        if url.endswith('/'):
            rp.set_url(url + 'robots.txt')
            robot_txt = url + 'robots.txt'
            rp.read()
        else:
            rp.set_url(url + '/robots.txt')
            robot_txt = url + '/robots.txt'
            rp.read()

        print("Permission to crawl " + str(url) + " : " + str(rp.can_fetch('*', robot_txt)))

        if rp.can_fetch('*', robot_txt):
            return True
        else:
            return False

    except Exception as e:
        print("Something went wrong reading the robots.txt file...")
        return False


# Takes in a URL and returns all hyperlink html tags "a"
def visit_url(url):
    try:
        req = Request(url)
        page = urlopen(req).read()
        soup = BeautifulSoup(page, 'html.parser')
        return soup.find_all('a')

    except Exception as e:
        print("Request failed")
        return []


# Extracts links from the provided URL for a total of n files
# Returns set of URLs visited
def extract_links(url, n):
    counter = 0
    visited_list = set()
    open_list = {url}

    while len(open_list) > 0 and counter < n:
        link = open_list.pop()
        print(link)
        new_links = visit_url(link)
        visited_list.add(link)

        counter = counter + 1

        for l in new_links:
            try:
                file_name = l['href']

                # Making sure only given host is scraped
                if file_name.startswith('http') and not file_name.startswith(url):
                    continue

                if file_name.startswith('mailto') or file_name.startswith('tel'):
                    continue

                url_new = file_name

                if not file_name.startswith(url):
                    url_new = url + file_name

                if url_new not in visited_list:
                    open_list.add(url_new)

            except Exception as e:
                pass

    return visited_list


# Takes in a set of URLs and returns their text content
def read_urls(urls):
    documents = []
    for url in urls:
        try:
            req = Request(url)
            page = urlopen(req).read()
            soup = BeautifulSoup(page, 'html.parser')
            [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]

            # Remove excessive white spaces from text
            doc = re.sub(r'\s+', ' ', soup.get_text())

            # Remove URLs from text
            doc = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', '', doc)
            documents.append(doc)

        except Exception as e:
            print("Something went wrong reading the URLs - " + str(e))
            pass

    # print(documents)
    print("Total amount of documents processed: " + str(len(documents)))

    return documents


# Takes list of documents to perform clustering and sentiment analysis operations on.
# k number of clusters
# 50 iterations
# Sentiment analysis for each cluster is calculated by taking the 15 most popular words in each cluster
# and averaging the sentiment score
def perform_k_means_clustering_and_sentiment_analysis(docs, k):
    afinn = Afinn()

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(docs)
    model = KMeans(n_clusters=k, init='k-means++', max_iter=50, n_init=1)
    model.fit(X)

    print("Top terms per cluster:")
    ordered_centroids = model.cluster_centers_.argsort()[:, ::-1]
    try:
        terms = vectorizer.get_feature_names()
    except Exception as e:
        pass
    for i in range(k):
        sentiment_score = 0

        print("Cluster %d:" % i + " 50 most popular words:")

        # Computing sentiment score for top 10% of most common words
        for index in ordered_centroids[i, :math.floor(len(ordered_centroids[i])/10)]:
            # Use for testing:
            # print('%s - %f'%(terms[index], afinn.score(terms[index])))
            print(terms[index])
            sentiment_score = sentiment_score + afinn.score(terms[index])

        print("Cluster " + str(i) + " sentiment score: " + str(sentiment_score))


if __name__ == '__main__':
    # Please ensure link is of format 'http(s)]://[host]'
    url = "https://concordia.ca"

    if can_crawl(url):
        docs = read_urls(extract_links(url, 100))
        perform_k_means_clustering_and_sentiment_analysis(docs, 6)
    else:
        print("Cannot crawl " + str(url) + ". Terminating ... ")
        exit()
