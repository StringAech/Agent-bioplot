#%% import the python packages
import scanpy as sc
import pandas as pd

#%% define the function to check the data
def inspect_data(adata):
    """
    Function to inspect the data in an AnnData object.
    
    Parameters:
    adata (AnnData): The AnnData object to inspect.
    
    Returns:
    dict: A dictionary containing information about the data.
    """
    
    # Check if the input is an AnnData object
    if not isinstance(adata, sc.AnnData):
        raise ValueError("Input must be an AnnData object.")
    
    # Get basic information about the data
    info = {
        "n_obs": adata.n_obs, # number of observations (cells)
        "n_vars": adata.n_vars, # number of variables (genes)
        "obs_colums": adata.obs.columns.tolist(), # list of observation columns
        "uns": list(adata.uns.keys()),
        "obsm": list(adata.obsm.keys()),
        "varm": list(adata.varm.keys()),
        "layers": list(adata.layers.keys()),
        "has_raw": adata.raw is not None,
    }
    return info

#%% define the function to inspect the adata.obs
def inspect_obs_column(adata: sc.AnnData, column: str, max_unique: int = 20) -> dict:
    """
    Inspect one column in adata.obs.

    Parameters
    ----------
    adata : AnnData
    column : str
    max_unique : int

    Returns
    -------
    dict
    """
    # Check if the input is an AnnData object
    # if not isinstance(adata, sc.AnnData):
    #     raise ValueError("Input must be an AnnData object.")
    
    # Maximum number must be greater than 0
    if max_unique <= 0:
        raise ValueError("max_unique must be greater than 0.")
    
    # Check if the column exists in adata.obs
    if column not in adata.obs.columns:
        raise ValueError(f"Column '{column}' does not exist in adata.obs.")
    
    # Get the column dtype
    dtype = adata.obs[column].dtype

    # Get the number of unique values in the column
    unique_value_counts = adata.obs[column].nunique()

    # Decide the next step based on the if the number of unique values is less than to max_unique
    if unique_value_counts <= max_unique:
        values = adata.obs[column].dropna().unique().tolist()
    else:
        values = list(adata.obs[column].value_counts().head(max_unique).index)
    
    dict_info = {
        "column": column,
        "dtype": str(dtype),
        "unique_value_counts": unique_value_counts,
        "values": values,
        "truncated": unique_value_counts > max_unique
    }
    
    return dict_info
