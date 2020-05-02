from time import time

import pandas as pd
import re

# Data Preprocessing class
# Allows for filtering rows and copying and deleting text within columns

class CleanData:
    def __init__(self, data='data/external/data_cleaned.csv'):
        self.df_data = pd.read_csv(data)
        self.processed_data = None

    def extract_column(self, col): # Returns desired column as a list
        return self.df_data[col].tolist()

    def todict(self): # Returns a dict object of the dateframe
        return self.df_data.to_dict()

    def search_(self, col, keywords): # Returns boolean dataframe of rows with keywords
        search_pattern = '|'.join(map(re.escape, keywords))
        return self.df_data[col].str.contains(search_pattern, na=False, case=False)

    def remove_rows_with(self, col, keywords=['case report']): 
        # Removes all rows where keywords are found
        search_results = self.search_(col, keywords)
        removed_rows = self.df_data.drop(self.df_data[~search_results].index)
        self.df_data.drop(self.df_data[search_results].index, inplace=True)
        return removed_rows

    def remove_rows_without(self, col, keywords=['methods:']): 
        # Removes all rows where keywords are not found
        search_results = self.search_(col, keywords)
        removed_rows = self.df_data.drop(self.df_data[search_results].index)
        self.df_data.drop(self.df_data[~search_results].index, inplace=True)
        return removed_rows

    def remove_columns(self,col): # Removes specified column
        self.df_data.drop(labels=col, axis=1, inplace=True)

    def copy_text(self, search_col, start_keywords=["methods:"], end_keywords=["results:"], dest_col='methods'):
        # Copies text between the first index postion of the first 'start_keywords' found 
        # and the first index position of the first 'end_keywords' found
        # If no end_end keyword found, then text copied until end of string
        new_col = []
        start_search_pattern = '|'.join(map(re.escape, start_keywords))
        end_search_pattern = '|'.join(map(re.escape, end_keywords))
        for row in self.df_data[search_col]:
            start = re.search(start_search_pattern, row, re.IGNORECASE)
            end = re.search(end_search_pattern, row, re.IGNORECASE)
            if start is None:
                start_index = start
            else:
                start_index = start.start()
            if end is None:
                end_index = end
            else:
                end_index = end.start()
            new_col.append(row[start_index:end_index])
        self.df_data[dest_col] = new_col

    def delete_text(self, search_col, dest_col, start_keywords=["methods:"], end_keywords=["conclusion:"]):
        # Deletes text between the first index postion of the first 'start_keywords' found 
        # and the first index position of the first 'end_keywords' found
        # If no end_end keyword found, then all text after 'start_keywords' is deleted
        new_col = []
        start_search_pattern = '|'.join(map(re.escape, start_keywords))
        end_search_pattern = '|'.join(map(re.escape, end_keywords))
        for row in self.df_data[search_col]:
            start = re.search(start_search_pattern, row, re.IGNORECASE)
            end = re.search(end_search_pattern, row, re.IGNORECASE)
            if start is None:
                start_index = start
            else:
                start_index = start.start()
            if end is None:
                end_index = end
            else:
                end_index = end.start()
            new_col.append(row[:start_index]+row[end_index:])
        self.df_data[dest_col] = new_col

    def save_data(self, file_path_name): # Saves data as CSV
        self.df_data.to_csv(file_path_name, index=False)
        print("Data saved to: " + file_path_name)

if __name__ == "__main__": # Prevents the following code from running when importing module
    #import dataset
    print("Loading dataset...")
    t0 = time()
    data = CleanData(data='data/external/data_cleaned.csv')
    print("done in %0.3fs." % (time() - t0))

    num_rows = len(data.df_data.index)
    print("Filtering %d rows..." % num_rows)
    t0 = time()
    df_filtered_out = data.remove_rows_with(col="title_abstract", keywords=["case report"])
    print("done in %0.3fs." % (time() - t0))
    new_num_rows = len(data.df_data.index)
    print("%d rows removed \n %d rows remain" % (num_rows - new_num_rows, new_num_rows ))
    data.save_data('data/external/data_filtered.csv')
    
    filtered_out_path = 'data/processed/data_casereport.csv'
    df_filtered_out.to_csv(filtered_out_path, index=False)
    print("Deleted data saved to: " + filtered_out_path)

    print("Filtering %d rows..." % len(data.df_data.index))
    t0 = time()
    df_filtered_out = data.remove_rows_without(col="title_abstract", keywords=["methods:","procedures:"])
    print("done in %0.3fs." % (time() - t0))
    new_num_rows = len(data.df_data.index)
    print("%d rows removed \n %d rows remain" % (num_rows - new_num_rows, new_num_rows ))

    filtered_out_path = 'data/processed/removed_data_no_methods.csv'
    df_filtered_out.to_csv(filtered_out_path, index=False)
    print("Deleted data saved to: " + filtered_out_path)

    print("Copying methods to new column...")
    t0 = time()
    data.copy_text(search_col = "title_abstract", 
                    dest_col='methods',
                    start_keywords=["methods:", "procedures:"], 
                    end_keywords=["results:", "findings:", "conclusion:", "conclusions:"], 
                    )
    print("done in %0.3fs." % (time() - t0))

    print("Deleting methods and results from origin column...")
    t0 = time()
    data.delete_text(search_col = "title_abstract", 
                    dest_col='title_abstract',
                    start_keywords=["methods:", "procedures:"], 
                    end_keywords=["conclusion:", "conclusions:"], 
                    )
    print("done in %0.3fs." % (time() - t0))
    data.save_data('data/external/data_methods_split.csv')
