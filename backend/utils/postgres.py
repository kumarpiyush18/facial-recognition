import os
import sys

sys.path.append(os.getcwd())

import psycopg2
import psycopg2.extras
from psycopg2._psycopg import AsIs
from utils.logger import get_logger

logger = get_logger(__name__)


class PostgresHandle:
    def __init__(self) -> None:
        self.user = "root"
        self.password = ""
        self.host = "127.0.0.1"
        self.database = "lending"
        self.port = "5432"
        self.conn = None
        self.cursor = None

        self.set_connection()

    def __del__(self):
        self.conn.close()

    def get_connection(self):
        if not self.conn or not self.cursor:
            print(f"postgres user: {self.user}, db_host: {self.host}, db_name: {self.database}")
            self.set_connection()

        return self.conn, self.cursor

    def set_connection(self):
        print(f"Setting postgres connection")
        print(f"postgres user: {self.user}, db_host: {self.host}, db_name: {self.database}")

        self.conn = psycopg2.connect(
            database=self.database, user=self.user, password=self.password, host=self.host, port=self.port
        )
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()
        print(f"Postgres connection ready!")

    def put(self, item, table):
        columns = item.keys()
        values = [item[column] for column in columns]
        insert_statement = f'insert into {table} (%s) values %s'
        self.cursor.execute(insert_statement, (AsIs(','.join(columns)), tuple(values)))

    def get_by_id(self, id, table):
        select_statement = f"select * from {table} where id={id}"
        self.cursor.execute(select_statement)
        columns = list(self.cursor.description)
        response = self.cursor.fetchall()
        results = []
        if response is None:
            return results
        for res in response:
            row = {}
            for i, col in enumerate(columns):
                row[col.name] = res[i]
            results.append(row)
        return results


if __name__ == '__main__':
    postgres = PostgresHandle()
    print(postgres.get_by_id(2, 'user_consent'))

