import numpy as np
import pandas as pd
from rpy2.robjects import globalenv, r, pandas2ri
pandas2ri.activate()

from statsmodels.regression.linear_model import (RegressionResults,
                                                 RegressionResultsWrapper)

from statsmodels.api import OLS

from scipy import stats


class RRegressionResults:
    """
    This holds results from a model estimated with R, but mimicks the API of
    one estimated with statsmodels.
    """
    # All work is done directly on instances.


class RRegressionModel(OLS):
    """
    Compatibility class to run models in R as if they were actually run with
    statsmodels.

    Only meant to be used through with_formula().
    """
    def __init__(self, *args, **kwargs):
        """
        This in fact is never run, as we create a real OLS object in
        "from_formula" below, and then change its __class__.
        """
        if len(args) or len(kwargs):
            raise NotImplementedError("This class is not meant to be used "
                                      "directly - please use from_formula()")

    @classmethod
    def from_formula(cls, formula, data, command='lm', libraries=[],
                     debug=False, **kwargs):
        """
        Estimate a model by passing a formula and data, in the spirit of
        statsmodels.api.OLS.from_formula().

        Additionally supports the following arguments:

        Parameters
        ----------
        command : string, default 'lm'
            R command used for the estimation.
        libraries : list-like of strings, default empty
            R libraries which should be loaded before the estimation.
        debug : Bool, default False
            If True, print debug messages.
        **kwargs : additional arguments
            Arguments to be passed to the R command.
        """

        # Creating the OLS object and only then hijacking it allows us to best
        # profit of statsmodels' machinery:
        mod = OLS.from_formula(formula, data)
        mod.__class__ = RRegressionModel

        # This holds stuff statsmodels is not aware of, and fit() needs:
        mod._backstage = {'libraries' : libraries, 'command' : command,
                          'full_data' : data, 'kwargs' : kwargs}
        mod._debug = (lambda *x : print(*x)) if debug else (lambda *x : None)
        return mod

    def fit(self):

        ## STEP 1: setup R environment and run the estimation

        for library in self._backstage['libraries']:
            r("library('{}')".format(library))

        globalenv['data'] = self._backstage['full_data']
        formula_templ = "res <- {}({}, data=data{})"

        # FIXME: process kwargs
        self._full_formula = formula_templ.format(self._backstage['command'],
                                                  self.formula, '')

        self._debug("Run", self._full_formula)
        r(self._full_formula)
        r("rsum <- summary(res)")

        ## STEP 2: retrieve all results we need

        res = r['rsum']
        self._debug(res)

        # Retrieve confidence intervals:
        ci = r("ci <- confint(res)")

        _attrs = self._inspect_R(res, ci=ci)

        ## STEP 3: package the results a statsmodels-like format

        # Sometimes features are retrieved from wrapper (stargazer does this),
        # other times from the actual result (statsmodels' summary_col does
        # this), so we'll have both.
        rres = RRegressionResults()
        rres.target = self.formula.split()[0]
        rres.model = self

        # We need to hijack this rather than subclassing because stargazer does
        # not use "isistance()" but "type()":
        wrap = RegressionResultsWrapper(rres)

        # All items except "params" are @cache_readonly and need first to be
        # deleted, and then redefined:
        for attr in _attrs:
            if attr not in 'params':
                if hasattr(rres, attr):
                    delattr(rres, attr)
            setattr(rres, attr, _attrs[attr])
            setattr(wrap, attr, _attrs[attr])

        rres.__class__ = RegressionResults

        # Clean up memory:
        r("gc()")

        return wrap

    def _inspect_R(self, res, ci=None):
        """
        Extract from an R result the various pieces.

        Parameters
        ----------
        res : R object
            R summary of a fitted model.
            Typically obtained with "summary(fitted)" (in R).
        ci : R object
            Confidence intervals of a fitted model
            Typically obtained with "confint(fitted)" (in R).
        """

        d_res = dict(zip(res.names, res))
        coeffs_mat = d_res['coefficients']

        # FIXME: there MUST be a better way, the results matrix in R KNOWS its
        # names. I shouldn't be retrieving them separately, and guessing the
        # correct columns:
        coef_names = r("names(coef(res))")

        # R denotes the intercept as "(Intercept)", statsmodels as "Intercept":
        intercept_idces = np.where(coef_names == '(Intercept)')[0]
        coef_names[intercept_idces[0]] = 'Intercept'

        # Retrieve main results:
        _attrs = {
        'params' : pd.Series(coeffs_mat[:,0], index=coef_names),
        'tvalues' : pd.Series(coeffs_mat[:,2], index=coef_names),
        'pvalues' : pd.Series(coeffs_mat[:,3], index=coef_names),
        'bse' : pd.Series(coeffs_mat[:,1], index=coef_names),
        'rsquared' : d_res['r.squared'][0],
        'rsquared_adj' : d_res['adj.r.squared'][0],
        'scale' : d_res['sigma'][0]**2
        }

        (_attrs['fvalue'],
         _attrs['df_model'],
         _attrs['df_resid']) = d_res['fstatistic']
        # Couldn't find this ready in the R summary
        _attrs['f_pvalue'] = stats.f.sf(_attrs['fvalue'],
                                        _attrs['df_model'],
                                        _attrs['df_resid'])

        if ci is not None:
            ci = pd.DataFrame(r['ci'], index=coef_names)

            def conf_int(alpha=0.05):
                if alpha != 0.05:
                    raise NotImplementedError("Only alpha=0.05 is supported, "
                                              "{} passed".format(alpha))
                return ci
            _attrs['conf_int'] = conf_int

        return _attrs
