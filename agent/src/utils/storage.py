import os
import csv

import pandas as pd


class Storage:

    def __init__(self, directory: str = "data", debug: bool = False):
        self._tables = dict()
        '''
        "users.csv" -> pd.DataFrame
        '''
        self._description = ""
        self._debug = debug

        self._read_directory(directory)

    def _add_table(self, table: pd.DataFrame, name: str):
        if self._debug:
            print(f"Adding table {name}")
        self._tables[name] = table

    def _add_description(self, description: str):
        if self._debug:
            print(f"Adding description")
        self._description = description


    def _read_directory(self, path: str):
        if self._debug:
            print(f"Reading directory {path}")
        for file in os.listdir(path):
            if file.endswith(".csv"):
                table = pd.read_csv(os.path.join(path, file), sep=self._detect_delimiter(os.path.join(path, file)))
                self._add_table(table, file)
            elif file == "readme.txt":
                with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                    data = f.read()
                self._add_description(data)
            else:
                print(f"Unknown file {file}")

    @staticmethod
    def _detect_delimiter(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(8192)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            return dialect.delimiter
        except csv.Error:
            return ","

    @property
    def description(self):
        return self._description

    @property
    def tables_headers(self):
        return [f"{table} ({', '.join(self._tables[table].columns)})" for table in self._tables]

    def get_columns(self, table_name: str) -> list[str]:
        table = self.get_table(table_name)
        return table.columns.tolist()

    def get_table(self, table_name: str):
        table = self._tables.get(table_name)
        if table is None:
            raise Exception(f"Table {table_name} not found")
        return table
