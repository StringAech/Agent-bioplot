#%% import the python packages
import scanpy as sc
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path

#%% define the function to check the data
def inspect_data(adata: sc.AnnData) -> dict:
    """
    Function to inspect the data in an AnnData object.
    
    Parameters:
    adata (AnnData): The AnnData object to inspect.
    
    Returns:
    dict: A dictionary containing information about the data.
    """

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

#%% define the function to infer column type
def infer_column_type(adata: sc.AnnData, column: str) -> str:
    """
    Infer the type of a column in adata.obs.

    Parameters
    ----------
    adata : AnnData
    column : str

    Returns
    -------
    str
        The inferred type of the column: 'categorical', 'numeric', or 'unknown'.
    """
    # Check if the column exists in adata.obs
    if column not in adata.obs.columns:
        raise ValueError(f"Column '{column}' does not exist in adata.obs.")
    
    # if the column is categorical or string, return 'categorical'.
    if isinstance(adata.obs[column].dtype, pd.CategoricalDtype):
        return 'categorical'
    
    # if the column is string, return 'categorical'.
    if pd.api.types.is_string_dtype(adata.obs[column]):
        return 'categorical'

    # if the column is numeric, return 'numeric'.
    if pd.api.types.is_numeric_dtype(adata.obs[column]):
        return 'numeric'
    
    # elseif return 'unknown'.
    return 'unknown'


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

    # Maximum number must be greater than 0
    if max_unique <= 0:
        raise ValueError("max_unique must be greater than 0.")
    
    # Check if the column exists in adata.obs
    if column not in adata.obs.columns:
        raise ValueError(f"Column '{column}' does not exist in adata.obs.")
    
    # Get the column dtype
    dtype = adata.obs[column].dtype

    # Get the column type using the infer_column_type function
    column_type = infer_column_type(adata, column)

    # Base on column type, decide the next step
    if column_type == 'categorical':
        # Get the number of unique values in the column
        unique_value_counts = adata.obs[column].nunique()

        # Decide the next step based on the if the number of unique values is less than to max_unique
        values = list(adata.obs[column].value_counts().head(max_unique).index)
        
        dict_info = {
            "column": column,
            "column_type": column_type,
            "dtype": str(dtype),
            "unique_value_counts": unique_value_counts,
            "values": values,
            "truncated": unique_value_counts > max_unique
        }
    elif column_type == 'numeric':
        min_value = float(adata.obs[column].min())
        max_value = float(adata.obs[column].max())
        mean_value = float(adata.obs[column].mean())
        median_value = float(adata.obs[column].median())
        n_missing_value = int(adata.obs[column].isna().sum())
        dict_info = {
            "column": column,
            "column_type": column_type,
            "dtype": str(dtype), 
            "min": min_value if not pd.isna(min_value) else None,
            "max": max_value if not pd.isna(max_value) else None,
            "mean": mean_value if not pd.isna(mean_value) else None,
            "median": median_value if not pd.isna(median_value) else None,
            "n_missing": n_missing_value
        }
    else:
        n_missing_value = int(adata.obs[column].isna().sum())
        dict_info = {
            "column": column,
            "column_type": column_type,
            "dtype": str(dtype),
            "n_missing": n_missing_value
        }
    return dict_info

#%% define the function generate the umap plot parameters
def umap_params(colors: str = 'leiden', point_size: float = 2) -> dict:
    dict_info = {
        "color_by": colors,
        "point_size": point_size
    }
    return dict_info

#%% define the functio to plot the umap
def plot_umap(adata: sc.AnnData, params: dict):
    """
    Plot UMAP using scanpy's pl.umap function.
    """

    # Check if the parameters is a dictionary.
    if not isinstance(params, dict):
        raise TypeError("Parameters must be a dictionary.")

    # Check if the point_size is a positive number.
    point_size = params.get("point_size", 2)
    if (
        isinstance(point_size, bool)
        or not isinstance(point_size, (int, float))
        or not math.isfinite(point_size)
        or point_size <= 0
    ):
        raise ValueError("point_size must be a positive number.")
    
    # Check if the color_by is a non-empty string.
    color_by = params.get("color_by")
    if not isinstance(color_by, str) or not color_by.strip():
        raise ValueError("colors must be a non-empty string.")
    
    # Check if the color_by is a valid column in adata.obs.
    if color_by not in adata.obs.columns:
        raise ValueError(f"Column '{color_by}' does not exist in adata.obs.")
    
    # Check if the X_umap is computed in adata.obsm.
    if "X_umap" not in adata.obsm:
        raise ValueError("UMAP coordinates are missing from adata.obsm.")
    
    # Plot the UMAP using scanpy's pl.umap function.
    plot = sc.pl.umap(adata, color=color_by, size=point_size, show=False, return_fig=True)
    plt.close(plot)

    return plot

#%% define the function to save the plot
def save_plot(fig: Figure, save_dir: Path, filename: str = 'umap', extension: str = 'png', dpi:int = 300) -> str:
    """
    Save the plot to a file.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure to save.
    save_dir : str
        The directory to save the plot.
    filename : str
        The name of the file (without extension).
    extension : str, optional
        The file extension (default is 'png').
    dpi : int, optional
        The resolution in dots per inch (default is 300).

    Returns
    -------
    str
        The full path to the saved file.
    """
    # Check the dpi is a positive integer.
    if not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer.")

    # Check the fig is a matplotlib Figure.
    if not isinstance(fig, Figure):
        raise TypeError("fig must be a matplotlib Figure.")

    # Check the extension is a string and is one of the supported formats.
    supported_extensions = ['png', 'jpg', 'jpeg', 'pdf', 'svg']
    if not isinstance(extension, str) or extension.lower() not in supported_extensions:
        raise ValueError(f"extension must be one of {supported_extensions}.")
    extension = extension.lower()

    # Build the full file path and check the save_dir is exists, if not create it.
    savedir_path = Path(save_dir)
    if not savedir_path.exists():
        savedir_path.mkdir(parents=True, exist_ok=True)
    
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string.")
    save_path = savedir_path / f"{filename}.{extension}"

    # Save the figure to the specified file.
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return str(save_path.absolute())