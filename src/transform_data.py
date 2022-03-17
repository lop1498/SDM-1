from neo4j import GraphDatabase
import numpy as np
import pandas as pd


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


conn = Neo4jConnection(uri="bolt://localhost:7687", user="neo4j", pwd="difficulties-pushup-gaps")


def add_authors(path):
    df = pd.read_csv(path, header=[0], nrows=50000)

    query = '''
            LOAD CSV
            UNWIND $rows AS row
            MERGE (c:Category {category: row.category})
            RETURN count(*) as total
            '''
    return conn.query(query, parameters = {'rows':categories.to_dict('records')})


def add_articles(path):
    df = pd.read_csv(path, header=[0], nrows=50000)

    query = '''
            LOAD CSV
            UNWIND $rows AS row
            MERGE (c:Category {category: row.category})
            RETURN count(*) as total
            '''
    return conn.query(query, parameters={'rows': categories.to_dict('records')})



def add_articles(path):
    n = pd.read_csv(path)
    df = pd.read_csv(path, header=[0], nrows=50000)

    query = '''
            LOAD CSV
            UNWIND $rows AS row
            MERGE (c:Category {category: row.category})
            RETURN count(*) as total
            '''
    return conn.query(query, parameters={'rows': categories.to_dict('records')})


if __name__ == "__main__":
    #file = pd.read_csv('/Users/lop1498/Desktop/MDS/Q2/SDM/lab1/data/dblp_article.csv')
    path = '/Users/lop1498/Desktop/MDS/Q2/SDM/lab1/data/'

    #conn.query('CREATE CONSTRAINT papers IF NOT EXISTS ON (p:Paper)     ASSERT p.id IS UNIQUE')
