import pandas as pd
from pathlib import Path
from typing import List, Tuple

import rpy2.robjects as robjects
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri


def load_rdata_rpy2(rdata_path: str) -> pd.DataFrame:
    """
    Loads an .RData file and extracts the 'metashort' dataframe from the list 'M'.

    Args:
        rdata_path: Path to the .RData file.

    Returns:
        A pandas DataFrame containing the 'metashort' data.

    Raises:
        ValueError: If 'M' or 'metashort' cannot be found or converted.
        TypeError: If the resulting object is not a pandas DataFrame.
    """
    rdata_path = Path(rdata_path).resolve()

    # Load base R package and create a new isolated environment
    base = importr('base')
    r_env = r['new.env']()

    # Load the RData file into the isolated environment
    base.load(str(rdata_path), envir=r_env)

    # Check if 'M' exists in the environment
    if not r['exists']('M', envir=r_env):
        raise ValueError("Object 'M' not found in the .RData file")

    M = r['get']('M', envir=r_env)

    try:
        # Access the 'metashort' element from the 'M' list
        metashort = M.rx2('metashort')

        # Convert the R dataframe to a pandas dataframe using localconverter
        with localconverter(default_converter + pandas2ri.converter):
            df = pandas2ri.rpy2py(metashort)

        if not isinstance(df, pd.DataFrame):
            raise TypeError("'metashort' is not a valid pandas DataFrame")

        return df

    except Exception as e:
        raise ValueError(f"Failed to access 'metashort' from 'M': {e}")


def downsample(data: List[Tuple[float, float]], threshold: int) -> List[Tuple[float, float]]:
    """
    Applies the LTTB (Largest-Triangle-Three-Buckets) algorithm to downsample a time series.

    Args:
        data: A list of (x, y) tuples representing the original data points.
        threshold: The number of points to reduce the dataset to.

    Returns:
        A list of (x, y) tuples representing the downsampled data.
    """
    if threshold >= len(data) or threshold <= 2:
        return data

    sampled = [data[0]]
    every = (len(data) - 2) / (threshold - 2)
    a = 0

    for i in range(threshold - 2):
        avg_range_start = int((i + 1) * every) + 1
        avg_range_end = int((i + 2) * every) + 1
        avg_range_end = min(avg_range_end, len(data))

        avg_x = avg_y = 0.0
        avg_range_len = avg_range_end - avg_range_start

        for idx in range(avg_range_start, avg_range_end):
            avg_x += data[idx][0]
            avg_y += data[idx][1]

        if avg_range_len > 0:
            avg_x /= avg_range_len
            avg_y /= avg_range_len

        range_offs = int(i * every) + 1
        range_to = int((i + 1) * every) + 1
        range_to = min(range_to, len(data))

        point_a_x, point_a_y = data[a]

        max_area = -1.0
        max_area_point = data[a]
        next_a = a

        for idx in range(range_offs, range_to):
            if idx >= len(data):
                break
            point = data[idx]
            area = abs(
                (point_a_x - avg_x) * (point[1] - point_a_y) -
                (point_a_x - point[0]) * (avg_y - point_a_y)
            ) * 0.5

            if area > max_area:
                max_area = area
                max_area_point = point
                next_a = idx

        sampled.append(max_area_point)
        a = next_a

    sampled.append(data[-1])
    return sampled
