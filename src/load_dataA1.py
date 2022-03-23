import random

from neo4j import GraphDatabase
import numpy as np
import pandas as pd
import urllib.parse
from neo4j import GraphDatabase
from sklearn.utils import shuffle
from random import randint, seed, sample, choice
from nltk.corpus import stopwords
import time


class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(self, query, parameters=None, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None:
                session.close()
        return response


conn = Neo4jConnection(uri="bolt://localhost:7687", user="neo4j", pwd="sdm")


def clean_database():
    query = "MATCH (n)-[r]-() DELETE r"
    conn.query(query)
    query = "MATCH (n) DELETE n"
    return conn.query(query)


def add_authors(path, path_db):
    # llegir els articles
    l = list(pd.read_csv(path + 'dblp_article_header.csv', sep=';').columns)
    names = [name.split(':')[0] for name in l]
    articles = pd.read_csv(path + 'dblp_article.csv', nrows=1500, sep=';', names=names)
    articles = articles[['article','volume','journal', 'author', 'title', 'mdate', 'key', 'year']].dropna()

    articles['author'] = articles['author'].map(lambda x: list(x.split('|')))
    articles = articles.explode('author')

    # llegir els autors
    df = pd.read_csv(path+'dblp_author.csv', header=[0], nrows=5000, sep=';')
    df.rename(columns={':ID':'id', 'author:string': 'author'}, inplace=True)

    # ens quedem amb els autors dels articles que tambe estiguin a authors
    df = pd.merge(df, articles, how='inner', on=['author'])
    df_aut = df.drop_duplicates(subset=['author'])
    df_aut.to_csv(path_db + "/authors.csv", index=False)
    p1 = "file:///authors.csv"
    df.to_csv(path_db+"/authors_edges.csv", index=False)
    p2 = "file:///authors_edges.csv"

    # generate dataframe with journal reviewers (minimum 3 maximum 6 per paper, not repeated)
    # take into account that an author of a paper can not be its own reviewer
    auths = list(df_aut['author'].unique())
    authors_list = []
    papers_list = []

    for name, group in articles.groupby('title'):
        auths_group = list(group['author'].unique())
        auths_possible = list(set(auths) - set(auths_group))
        for i in range(np.random.randint(3, 4)):
            authors_list.append(auths_possible[i])
            papers_list.append(name)

    df_reviews = pd.DataFrame({'title': papers_list, 'author': authors_list})
    df_reviews.to_csv(path_db + "/papers_reviews_journal.csv", index=False)
    p3 = "file:///papers_reviews_journal.csv"

    query1 = '''
            LOAD CSV WITH HEADERS FROM $p1 AS line
            CREATE(a:Author {id: line.id, name: line.author})
            '''

    query2 = '''
            LOAD CSV WITH HEADERS FROM $p2 AS line2
            MATCH (a:Author {id: line2.id}), (art:Article {title: line2.title})
            CREATE (a)-[r:writes_article]->(art)
            '''

    query3 = '''
            LOAD CSV WITH HEADERS FROM $p3 AS line3
            MATCH (a:Author {name: line3.author}), (art:Article {title: line3.title})
            MERGE (a)-[:writes_review]->(art)
           '''

    conn.query(query1, parameters={'p1': p1})
    conn.query(query2, parameters={'p2': p2})
    conn.query(query3, parameters={'p3': p3})

    return


def add_articles(path, path_db):
    l = list(pd.read_csv(path+'dblp_article_header.csv', sep=';').columns)
    names = [name.split(':')[0] for name in l]
    df = pd.read_csv(path+'dblp_article.csv', nrows=1500, sep=';', names=names)
    df = df[['article','volume','journal', 'author', 'title', 'mdate', 'key', 'year', 'author']].dropna()

    # generate a dataframe with minimum 1 and maximum 5 articles cited (avoiding autociting) for each article
    article_orig = []
    article_cited = []
    articles = list(df['title'].unique())

    for article in df['title'].unique():
        article_pos = list(set(articles) - {article})
        for i in range(np.random.randint(5, 12)):
            article_orig.append(article)
            article_cited.append(random.choice(article_pos))
    df_a = pd.DataFrame({'article_orig': article_orig, 'article_cited': article_cited})
    df_a.to_csv(path_db + "/articles_cite.csv", index=False)
    df.to_csv(path_db + "/articles.csv", index=False)

    # Papers cite articles
    df = pd.read_csv(path_db + '/papers.csv')
    papers = list(df['title'].unique())


    paper_orig = []
    paper_cited = []
    
    for paper in df['title'].unique():
        for i in range(np.random.randint(2, 7)):
            paper_orig.append(paper)
            paper_cited.append(random.choice(articles))
    df_p = pd.DataFrame({'paper_orig': paper_orig, 'article_cited': paper_cited})
    df_p.to_csv(path_db + "/papers_cite_article.csv", index=False)

    # Article cites paper
    paper_orig = []
    paper_cited = []

    for article in articles:
        for i in range(np.random.randint(2, 7)):
            paper_orig.append(article)
            paper_cited.append(random.choice(papers))
    df_p = pd.DataFrame({'article_orig': paper_orig, 'paper_cited': paper_cited})
    df_p.to_csv(path_db + "/article_cite_paper.csv", index=False)


    p1 = "file:///articles.csv"

    df = pd.read_csv(path+'dblp_article.csv', nrows=1500, sep=';', names=names)
    df = df[['article','volume','journal', 'author', 'title', 'mdate', 'key', 'year', 'author']].dropna()
    
    volumes = df.drop_duplicates(subset=['volume'])

    years = []

    for _ in range(len(volumes)):
        years.append(randint(2015,2019))

    volumes['year'] = years

    volumes.to_csv(path_db + "/volumes.csv", index=False)
    
    p2 = "file:///volumes.csv"

    p3 = "file:///articles_cite.csv"
    p4 = "file:///papers_cite_article.csv"
    p5 = "file:///article_cite_paper.csv"

    query1 = '''
                LOAD CSV WITH HEADERS FROM $p1 AS line1
                CREATE(:Article {key: line1.key, date: line1.mdate, title: line1.title})
                '''

    query2 = '''
                LOAD CSV WITH HEADERS FROM $p2 AS line2
                CREATE(j:Journal {name: line2.journal, year: line2.year})
                WITH j, line2
                MATCH (a:Article {key: line2.key})
                MERGE (a)-[r:published_in {volume: line2.volume}]->(j)
                '''

    query3 = '''
            LOAD CSV WITH HEADERS FROM $p3 AS line
            MATCH (a:Article {title: line.article_orig}), (b:Article {title: line.article_cited})
            CREATE (a)-[c:cites]->(b)
            '''
    query4 = '''
            LOAD CSV WITH HEADERS FROM $p4 AS line
            MATCH (a:Paper {title: line.paper_orig}), (b:Article {title: line.article_cited})
            MERGE (a)-[:cites]->(b)
            '''
    query5 = '''
            LOAD CSV WITH HEADERS FROM $p5 AS line
            MATCH (a:Article {title: line.article_orig}), (b:Paper {title: line.paper_cited})
            MERGE (a)-[:cites]->(b)
            '''

    conn.query(query1, parameters={'p1': p1})
    conn.query(query2, parameters={'p2': p2})
    conn.query(query3, parameters={'p3': p3})
    conn.query(query4, parameters={'p4': p4})
    conn.query(query5, parameters={'p5': p5})

    return


def add_papers(path, path_db):
    l = list(pd.read_csv(path + 'dblp_phdthesis_header.csv', sep=';').columns)
    names = [name.split(':')[0] for name in l]
    df = pd.read_csv(path + 'dblp_phdthesis.csv', nrows=1500, sep=';', names=names)
    df = df[['phdthesis', 'author', 'title', 'mdate', 'key', 'year', 'author']].dropna()

    stop = stopwords.words('english') + stopwords.words('spanish') + stopwords.words('german') + stopwords.words('portuguese') + ['-','.']
    l_abstract = list(df['title'])
    i = 0

    for abs in l_abstract:
        low = abs.lower()
        spl = low.split()
        resultwords  = [word for word in spl if word.lower() not in stop]
        result = ' '.join(resultwords)
        l_abstract[i] = result
        i += 1

    df.insert(loc=len(df.columns), column='abstract', value=l_abstract)

    df.to_csv(path_db + "/papers.csv", index=False)
    p1 = "file:///papers.csv"

    # generate a dataframe with minimum 1 and maximum 5 papers cited (avoiding autociting) for each paper
    paper_orig = []
    paper_cited = []
    papers = list(df['title'].unique())
    for paper in df['title'].unique():
        papers_pos = list(set(papers) - {paper})
        for i in range(np.random.randint(5, 20)):
            paper_orig.append(paper)
            paper_cited.append(random.choice(papers_pos))
    df_p = pd.DataFrame({'paper_orig': paper_orig, 'paper_cited': paper_cited})
    df_p.to_csv(path_db + "/papers_cite.csv", index=False)
    p2 = "file:///papers_cite.csv"

    query1 = '''
            LOAD CSV WITH HEADERS FROM $p1 AS line1
            CREATE(:Paper {key: line1.key, date: line1.mdate, title: line1.title, abstract: line1.abstract})
            '''

    query2 = '''
            LOAD CSV WITH HEADERS FROM $p2 AS line
            MATCH (a:Paper {title: line.paper_orig}), (b:Paper {title: line.paper_cited})
            MERGE (a)-[:cites]->(b)
            '''

    conn.query(query1, parameters={'p1': p1})
    conn.query(query2, parameters={'p2': p2})


def add_papers_authors(path, path_db):
    # read papers (phdthesis)
    l = list(pd.read_csv(path + 'dblp_phdthesis_header.csv', sep=';').columns)
    names = [name.split(':')[0] for name in l]
    papers = pd.read_csv(path + 'dblp_phdthesis.csv', nrows=5000, sep=';', names=names)
    papers = papers[['phdthesis', 'volume', 'author', 'title', 'mdate', 'key', 'year']].dropna()

    # explode papers dataframe by names
    papers['author'] = papers['author'].map(lambda x: list(x.split('|')))
    papers = papers.explode('author')
    papers.to_csv(path_db + "/papers_authors_edges.csv", index=False)
    p1 = "file:///papers_authors_edges.csv"

    # read authors names (that are related to articles, not papers)
    authors = pd.read_csv(path+'dblp_author.csv', header=[0], nrows=5000, sep=';')
    authors.rename(columns={':ID':'id', 'author:string': 'author'}, inplace=True)

    # cross product between papers authors and articles authors to mix both subsets
    conc = pd.merge(authors, papers, how='cross').sample(frac=0.013)
    conc.to_csv(path_db + "/papers_authors_more_edges.csv", index=False)
    p2 = "file:///papers_authors_more_edges.csv"

    # generate dataframe with reviewers (minimum 3 maximum 6 per paper, not repeated)
    # take into account that an author of a paper can not be its own reviewer
    auths = list(set(list(conc['author_x'].unique())+list(conc['author_y'].unique())))
    authors_list = []
    papers_list = []

    for name,group in conc.groupby('title'):
        auths_group = list(group['author_x'].unique() + list(group['author_y'].unique()))
        auths_possible = list(set(auths)-set(auths_group))
        for i in range(np.random.randint(3,6)):
            authors_list.append(auths_possible[i])
            papers_list.append(name)

    df_reviews = pd.DataFrame({'paper':papers_list, 'author': authors_list})
    df_reviews.to_csv(path_db + "/papers_reviews.csv", index=False)
    p3 = "file:///papers_reviews.csv"

    # queries
    query1 = '''
            LOAD CSV WITH HEADERS FROM $p1 AS line
            MERGE(a:Author {id: line.phdthesis, name: line.author})
            '''

    query2 = '''
            LOAD CSV WITH HEADERS FROM $p1 AS line2
            MATCH (a:Author {id: line2.phdthesis}), (pap:Paper {title: line2.title})
            CREATE (a)-[r:writes_paper]->(pap)
            CREATE (pap)-[c:corresponding_author]->(a)
            '''

    query3 = '''
            LOAD CSV WITH HEADERS FROM $p2 AS line3
            MERGE (a:Author {id: line3.id, name: line3.author_x})
            WITH a, line3
            MATCH (a:Author {id: line3.id}), (pap:Paper {key: line3.key})
            MERGE (a)-[:writes_paper]->(pap)
            '''

    query4 = '''
            LOAD CSV WITH HEADERS FROM $p3 AS line3
            MATCH (a:Author {id: line3.id}), (pap:Paper {title: line3.paper})
            MERGE (a)-[:writes_review]->(pap)
           '''

    conn.query(query1, parameters={'p1': p1})
    conn.query(query2, parameters={'p1': p1})
    conn.query(query3, parameters={'p2': p2})
    conn.query(query4, parameters={'p3': p3})

    return


def add_conferences(path, path_db):
    df = pd.read_csv(path+'conferences.csv')
    df.to_csv(path_db + "/conferences.csv", index=False)
    p1 = "file:///conferences.csv"

    query1 = '''
                LOAD CSV WITH HEADERS FROM $p1 AS line
                CREATE(c:Conference {name:line.conference})
                '''

    query2 = '''
                LOAD CSV WITH HEADERS FROM $p1 AS line
                MATCH (con:Conference {name: line.conference}), (ci:City {name: line.city})
                CREATE (con)-[r:held_in]->(ci)
                '''

    conn.query(query1, parameters={'p1': p1})
    conn.query(query2, parameters={'p1': p1})

    return


def add_edge_papers_to_conference(path, path_db):
    df_conf = pd.read_csv(path + 'conferences.csv')
    df_papers=pd.read_csv(path_db+'/papers.csv')

    lpapers = list(df_papers['title'])
    lconf = list(df_conf['conference'])
    n = len(lpapers)

    editions = []
    year = []
    conference_assignment = []

    for i in range(n):
        editions.append(randint(1,8))
        year.append(randint(2017,2023))
        conference_assignment.append(lconf[randint(0,len(lconf)-1)])

    data = {'Paper':lpapers, 'Conference':conference_assignment, 'Edition':editions, 'Year':year}
    df = pd.DataFrame(data)
    df.to_csv(path_db + "/papers_and_conferences.csv", index=False)

    p1 = "file:///papers_and_conferences.csv"

    query1 = '''
                LOAD CSV WITH HEADERS FROM $p1 AS line
                MATCH (pap:Paper {title: line.Paper}), (co:Conference {name: line.Conference})
                CREATE (pap)-[r:published_in {Edition: line.Edition, Year: line.Year}]->(co)
                '''

    conn.query(query1, parameters={'p1': p1})

    return


def add_topics(path, path_db):
    df_papers = pd.read_csv(path_db+'/papers.csv', nrows=5000)
    lpapers = list(df_papers['title'])
    keywords = list(df_papers['abstract'])
    i = 0

    for abs in keywords:
        w = abs.split()
        keys = sample(w, min(len(w)-1, randint(2,3)))

        keywords[i] = ':'.join(keys)
        i += 1

    df_topics = pd.read_csv(path+'topics.csv')
    ltopics = list(df_topics['Topics'])

    topics = [None] * 5000

    for i in range(len(topics)):
        topics[i] = choice(ltopics)

    df = pd.DataFrame({'paper': lpapers, 'keywords' : keywords, 'topics' : topics})
    df.to_csv(path + "paper_key_topic.csv", index=False)

    p1 = "file:///topics.csv"
    p2 = "file:///paper_key_topic.csv"

    query1 = '''
                LOAD CSV WITH HEADERS FROM $p1 AS line
                CREATE(t:Topic {name:line.Topics})
            '''

    query2 = '''
                LOAD CSV WITH HEADERS FROM $p2 AS line
                MATCH (pap:Paper {title: line.paper}), (to:Topic {name: line.topics})
                CREATE (pap)-[r:about {keywords:split(coalesce(line.keywords,""), ":")}]->(to)
            '''

    conn.query(query1, parameters={'p1': p1})
    conn.query(query2, parameters={'p2': p2})

    return


def add_cities(path, path_db):
    df = pd.read_csv(path + 'worldcities.csv')

    df.to_csv(path_db + "/cities.csv", index=False)
    p1 = "file:///cities.csv"

    query1 = '''
                LOAD CSV WITH HEADERS FROM $p1 AS line
                CREATE(c:City {name:line.city})
                '''

    conn.query(query1, parameters={'p1': p1})
    return


if __name__ == "__main__":
    paths_file = open('paths.txt', 'r')
    path = paths_file.readline()[:-1] # Porque lee el \n del salto de linea
    path_db = paths_file.readline()
    # path = '/Users/lop1498/Desktop/MDS/Q2/SDM/lab1/data/'
    # path_db = '/Users/lop1498/Library/Application Support/Neo4j Desktop/Application/relate-data/dbmss/dbms-a925e2f5-b2ac-42e3-b89e-1ea1b962a96b/import'
    start_time = time.time()

    clean_database()
    add_cities(path, path_db)
    add_conferences(path, path_db)
    add_papers(path, path_db)
    add_articles(path, path_db)
    # add_authors(path, path_db)
    # add_papers_authors(path, path_db)
    # add_edge_papers_to_conference(path,path_db)
    # add_topics(path, path_db)

    print("--- %s seconds ---" % (time.time() - start_time))
