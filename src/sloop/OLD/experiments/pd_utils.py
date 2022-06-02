

def flatten_column_names(df):
    """Flatten multi-index columns"""
    df.columns = list(map("-".join, df.columns.values))
    df.reset_index(inplace=True)
