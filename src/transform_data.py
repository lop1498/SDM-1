from neo4j import GraphDatabase
import numpy as np
import pandas as pd
import urllib.parse
from sklearn.utils import shuffle


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
    articles = pd.read_csv(path + 'dblp_article.csv', nrows=10000, sep=';', names=names)
    articles = articles[['article','volume','journal', 'author', 'title', 'mdate', 'key', 'year']].dropna()

    articles['author'] = articles['author'].map(lambda x: list(x.split('|')))
    articles = articles.explode('author')

    # llegir els autors
    df = pd.read_csv(path+'dblp_author.csv', header=[0], nrows=10000, sep=';')
    df.rename(columns={':ID':'id', 'author:string': 'author'}, inplace=True)

    # ens quedem amb els autors dels articles que tambe estiguin a authors
    df = pd.merge(df, articles, how='inner', on=['author'])
    df_aut = df.drop_duplicates(subset=['author'])
    df_aut.to_csv(path_db + "/authors.csv", index=False)
    p1 = "file:///authors.csv"
    df.to_csv(path_db+"/authors_edges.csv", index=False)
    p2 = "file:///authors_edges.csv"

    query1 = '''
            LOAD CSV WITH HEADERS FROM $p1 AS line
            CREATE(a:Author {id: line.id, name: line.author})
            '''

    query2 = '''
            LOAD CSV WITH HEADERS FROM $p2 AS line2
            MATCH (a:Author {id: line2.id}), (art:Article {title: line2.title})
            CREATE (a)-[r:writes_article]->(art)
            '''
    conn.query(query1, parameters={'p1': p1})
    conn.query(query2, parameters={'p2': p2})

    return


def add_articles(path, path_db):
    l = list(pd.read_csv(path+'dblp_article_header.csv', sep=';').columns)
    names = [name.split(':')[0] for name in l]
    df = pd.read_csv(path+'dblp_article.csv', nrows=10000, sep=';', names=names)
    df = df[['article','volume','journal', 'author', 'title', 'mdate', 'key', 'year', 'author']].dropna()

    df.to_csv(path_db + "/articles.csv", index=False)
    p1 = "file:///articles.csv"

    volumes = df.drop_duplicates(subset=['volume'])
    volumes.to_csv(path_db + "/volumes.csv", index=False)
    p2 = "file:///volumes.csv"

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

    conn.query(query1, parameters={'p1': p1})
    conn.query(query2, parameters={'p2': p2})

    return


def add_papers(path, path_db):
    l = list(pd.read_csv(path + 'dblp_phdthesis_header.csv', sep=';').columns)
    names = [name.split(':')[0] for name in l]
    df = pd.read_csv(path + 'dblp_phdthesis.csv', nrows=10000, sep=';', names=names)
    df = df[['phdthesis', 'author', 'title', 'mdate', 'key', 'year', 'author']].dropna()

    df.to_csv(path_db + "/papers.csv", index=False)
    p1 = "file:///papers.csv"

    query1 = '''
                    LOAD CSV WITH HEADERS FROM $p1 AS line1
                    CREATE(:Paper {key: line1.key, date: line1.mdate, title: line1.title})
                    '''

    conn.query(query1, parameters={'p1': p1})


def add_papers_authors(path, path_db):
    # llegir els papers
    l = list(pd.read_csv(path + 'dblp_phdthesis_header.csv', sep=';').columns)
    names = [name.split(':')[0] for name in l]
    papers = pd.read_csv(path + 'dblp_phdthesis.csv', nrows=10000, sep=';', names=names)
    papers = papers[['phdthesis', 'volume', 'author', 'title', 'mdate', 'key', 'year']].dropna()

    papers['author'] = papers['author'].map(lambda x: list(x.split('|')))
    papers = papers.explode('author')

    papers.to_csv(path_db + "/papers_authors_edges.csv", index=False)
    p1 = "file:///papers_authors_edges.csv"

    authors = pd.read_csv(path+'dblp_author.csv', header=[0], nrows=10000, sep=';')
    authors.rename(columns={':ID':'id', 'author:string': 'author'}, inplace=True)

    authors['_tmpkey'] = 1
    papers['_tmpkey'] = 1
    conc = pd.merge(authors, papers, on='_tmpkey').drop('_tmpkey', axis=1).sample(frac=0.0025)

    conc.to_csv(path_db + "/papers_authors_more_edges.csv", index=False)
    p2 = "file:///papers_authors_more_edges.csv"

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

    conn.query(query1, parameters={'p1': p1})
    conn.query(query2, parameters={'p1': p1})
    conn.query(query3, parameters={'p2': p2})

    return


if __name__ == "__main__":
    #paths_file = open('src/paths.txt', 'r')
    #path = paths_file.readline()[:-1] # Porque lee el \n del salto de linea
    #path_db = paths_file.readline()
    path = '/Users/lop1498/Desktop/MDS/Q2/SDM/lab1/data/'
    path_db = '/Users/lop1498/Library/Application Support/Neo4j Desktop/Application/relate-data/dbmss/dbms-a925e2f5-b2ac-42e3-b89e-1ea1b962a96b/import'

    clean_database()
    add_articles(path, path_db)
    add_authors(path, path_db)
    add_papers(path, path_db)
    add_papers_authors(path, path_db)
