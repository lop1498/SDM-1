from neo4j import GraphDatabase
import numpy as np
import pandas as pd
import urllib.parse


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
    df = pd.read_csv(path+'dblp_author.csv', header=[0], nrows=50000, sep=';')
    df.rename(columns={':ID':'id', 'author:string': 'author'}, inplace=True)

    df.to_csv(path_db+"/authors.csv", index=False)
    p = "file:///authors.csv"
    query = '''
            LOAD CSV WITH HEADERS FROM $p AS line
            CREATE(:Author {identifier: line.id, name: line.author})
            '''

    return conn.query(query, parameters={'p': p})


def add_articles(path, path_db):
    l = list(pd.read_csv(path+'dblp_article_header.csv', sep=';').columns)
    names = [name.split(':')[0] for name in l]
    df = pd.read_csv(path+'dblp_article.csv', nrows=50000, sep=';', names=names)

    df.to_csv(path_db + "/articles.csv", index=False)
    p = "file:///articles.csv"

    # attributes: key, mdate, title
    # relations: list of authors
    query = '''
                LOAD CSV WITH HEADERS FROM $p AS line
                CREATE(:Article {key: line.key, date: line.mdate, title: line.title})
                '''

    return conn.query(query, parameters={'p': p})


if __name__ == "__main__":
    path = '/Users/lop1498/Desktop/MDS/Q2/SDM/lab1/data/'
    path_db = '/Users/lop1498/Library/Application Support/Neo4j Desktop/Application/relate-data/dbmss/dbms-a925e2f5-b2ac-42e3-b89e-1ea1b962a96b/import'

    clean_database()
    add_authors(path, path_db)
    add_articles(path, path_db)
