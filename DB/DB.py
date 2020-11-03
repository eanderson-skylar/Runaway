import pandas as pd
import sqlalchemy

class DbOperation:
    def __init__(self):
        print('Creating database connection')
        config = open("../config/sql.txt", "r")
        self.engine = sqlalchemy.create_engine(config.read(), echo=False)

    def query_table(self, sql_query=None, table_name=None):
        if sql_query:
            print('Reading dataframe data from {}'.format(sql_query))
            df = pd.read_sql(sql_query, con=self.engine)
        elif table_name:
            print('Reading data from table: {}'.format(table_name))
            df = pd.read_sql_table(table_name=table_name, con=self.engine)

        print(df.count())
        return df

    def save_dataframe(self, df, table_name, mode='append'):
        print('Saving dataframe data to {}, size {}x{}'.format(table_name, df.shape[0], df.shape[1]))
        df.to_sql(name=table_name, con=self.engine, if_exists=mode, index=False, chunksize=1000)
        print('Data saved to database')