import pandas as pd
from rpy2.robjects import r
from rpy2.robjects import globalenv, r, pandas2ri
pandas2ri.activate()


def _df_from_r(rname):
    """
    Fetch a matrix from R with its labels.
    There are solutions for dataframes in rpy2
    https://stackoverflow.com/questions/20630121/
    ... but they don't work with the matrix of results from a model.

    Parameters
    ----------
    rname : R object name, or formula
        Name of object in the R environment, or formula to retrieve the desired
        matrix.
    """
    r("vtemp <- {}".format(rname))
    values = r['vtemp']
    kwargs = {}
    for what, how in ('index', 'rownames'), ('columns', 'colnames'):
        r("atemp <- {}(vtemp)".format(how))
        kwargs[what] = r['atemp']
    r("rm(vtemp)")
    r("rm(atemp)")
    return pd.DataFrame(values, **kwargs)
