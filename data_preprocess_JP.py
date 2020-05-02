#%%
# Load modules
from data_preprocessing import CleanData
from time import time

#%%
#import dataset
print("Loading dataset...")
t0 = time()
data = CleanData(data='data/external/data_cleaned.csv')
print("done in %0.3fs." % (time() - t0))

#%%
# Filter out "case report" string and save to data_filtered.csv
# Deleted data saved to data_casereport.csv
num_rows = len(data.df_data.index)
if False:
    print("Filtering %d rows..." % num_rows)
    t0 = time()
    df_filtered_out = data.remove_rows_with(col="title_abstract", keywords=["case report"])
    print("done in %0.3fs." % (time() - t0))
    new_num_rows = len(data.df_data.index)
    print("%d rows removed \n %d rows remain" % (num_rows - new_num_rows, new_num_rows ))
    data.save_data('data/processed/data_filtered.csv')
    filtered_out_path = 'data/processed/data_casereport.csv'
    df_filtered_out.to_csv(filtered_out_path, index=False)
    print("Deleted data saved to: " + filtered_out_path)

#%%
# Filter out all rows without a clearly identifiable methods section
# Deleted data saved to removed_data_no_methods.csv
print("Filtering %d rows..." % len(data.df_data.index))
t0 = time()
df_filtered_out = data.remove_rows_without(col="title_abstract", keywords=["methods:","procedures:"])
print("done in %0.3fs." % (time() - t0))
new_num_rows = len(data.df_data.index)
print("%d rows removed \n %d rows remain" % (num_rows - new_num_rows, new_num_rows ))
filtered_out_path = 'data/processed/removed_data_no_methods.csv'
df_filtered_out.to_csv(filtered_out_path, index=False)
print("Deleted data saved to: " + filtered_out_path)

#%%
# Copy methods data from "title_abstract" to a new "methods" column
print("Copying methods to new column...")
t0 = time()
data.copy_text(search_col = "title_abstract", 
                dest_col='methods',
                start_keywords=["methods:", "procedures:"], 
                end_keywords=["results:", "findings:", "conclusion:", "conclusions:"], 
                )
print("done in %0.3fs." % (time() - t0))

#%%
# Delete methods and results data from title_abstract column
print("Deleting methods and results from origin column...")
t0 = time()
data.delete_text(search_col = "title_abstract", 
                dest_col='title_abstract',
                start_keywords=["methods:", "procedures:"], 
                end_keywords=["conclusion:", "conclusions:"], 
                )
print("done in %0.3fs." % (time() - t0))

#%%
# Save final result to data_methods_split
data.save_data('data/processed/data_methods_split.csv')