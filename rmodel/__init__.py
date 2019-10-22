import numpy as np
import pandas as pd
from rpy2.robjects import globalenv, r, pandas2ri
pandas2ri.activate()

from statsmodels.regression.linear_model import (RegressionResults,
                                                 RegressionResultsWrapper)

from statsmodels.api import OLS

from scipy import stats
from patsy import ModelDesc

from .fake_number import FakeNumber

class RRegressionResults:
    """
    This holds results from a model estimated with R, but mimicks the API of
    one estimated with statsmodels.
    """
    # All work is done directly on instances.


class RModel(OLS):
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
        mod.__class__ = RModel

        if len(kwargs):
            raise NotImplementedError("Passing keyword arguments is still "
                                      "TODO.")

        # This holds stuff statsmodels is not aware of, and fit() needs:
        mod._backstage = {'libraries' : libraries, 'command' : command,
                          'full_data' : data, 'kwargs' : kwargs}
        mod._debug = (lambda *x : print(*x)) if debug else (lambda *x : None)
        return mod

    @classmethod
    def from_r_object(cls, rsum, ci=None):
        """
        Reconstruct a model from an rpy2 summary object, and optionally its
        confidence intervals.
        These can be easily saved in R with
            save(objname, file=file_name)
        and loaded in Python via rpy2 with
            r['load'](file_name)['objname']

        Parameters
        ----------
        rsum : R object
            R summary of a fitted model.
            Typically produced with "summary(fitted)" (in R).
        ci : R object
            Confidence intervals of the fitted model
            Typically produced with "confint(fitted)" (in R).
        """

        d_res = cls._r_as_dict(None, rsum)
        formula = str(d_res['terms']).splitlines()[0]

        # We want to create a fake dataset, and we use patsy to get the list of
        # variables. We are actually creating columns for interactions and
        # functions too... but who cares, identifying them would be at the
        # moment overkill.
        fobj = ModelDesc.from_formula(formula)
        # Remove Intercept, which is the first:
        varnames = [t.name()
                    for t in fobj.rhs_termlist + fobj.lhs_termlist][1:]
        data = pd.DataFrame(-1, index=[0], columns=varnames)

        # The even ugliest alternative to rescanning the formla like this was
        # to retrieve the formula from the first line of the representation of
        # the 'variables' attribute of d_res['terms'] as retrieved from R...

        # Creating the OLS object and only then hijacking it allows us to best
        # profit of statsmodels' machinery:
        mod = OLS.from_formula(formula, data)
        mod.__class__ = RModel

        attrs = mod._inspect_R(rsum, ci=ci)
        wrap = mod._package_attrs(attrs)
        
        return wrap

    def fit(self):

        ## STEP 1: setup R environment and run the estimation

        for library in self._backstage['libraries']:
            r("library('{}')".format(library))

        globalenv['data'] = self._backstage['full_data']
        formula_templ = "res <- {}({}, data=data{})"

        # TODO: process kwargs

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

        attrs = self._inspect_R(res, ci=ci)

        ## STEP 3: package the results a statsmodels-like format
        wrap = self._package_attrs(attrs)

        # Clean up memory:
        r("gc()")

        return wrap
        
    def _r_as_dict(self, robj):
        if isinstance(robj, dict):
            # Already dictified
            return robj

        d_res = dict(zip(robj.names, robj))
        return d_res

    def _inspect_R(self, res, ci=None):
        """
        Extract from an R result the various pieces.

        Parameters
        ----------
        res : R object or dict
            R summary of a fitted model: can be transformed to dict with
            RModel._r_to_dict(res).
            Typically produced with "summary(fitted)" (in R).
        ci : R object
            Confidence intervals of a fitted model
            Typically produced with "confint(fitted)" (in R).
        """

        d_res = self._r_as_dict(res)

        coeffs_mat = d_res['coefficients']

        # FIXME: there MUST be a better way, the results matrix in R KNOWS its
        # names. I shouldn't be retrieving them separately, and guessing the
        # correct columns:
        coef_names = r("names(coef(res))")

        # R denotes the intercept as "(Intercept)", statsmodels as "Intercept":
        intercept_idces = np.where(coef_names == '(Intercept)')[0]
        coef_names[intercept_idces[0]] = 'Intercept'

        # Retrieve main results:
        attrs = {
        'params' : pd.Series(coeffs_mat[:,0], index=coef_names),
        'tvalues' : pd.Series(coeffs_mat[:,2], index=coef_names),
        'pvalues' : pd.Series(coeffs_mat[:,3], index=coef_names),
        'bse' : pd.Series(coeffs_mat[:,1], index=coef_names),
        'rsquared' : d_res['r.squared'][0],
        'rsquared_adj' : d_res['adj.r.squared'][0],
        'scale' : d_res['sigma'][0]**2
        }

        (attrs['fvalue'],
         attrs['df_model'],
         attrs['df_resid']) = d_res['fstatistic']
        # Couldn't find this ready in the R summary
        attrs['f_pvalue'] = stats.f.sf(attrs['fvalue'],
                                       attrs['df_model'],
                                       attrs['df_resid'])

        if ci is None:
            msg = ("Trying to access the confidence intervals of a RModel "
                   "which wasn't passed any.")
            ci = pd.DataFrame(FakeNumber(msg),
                              index=coef_names, columns=range(2))
        else:
            ci = pd.DataFrame(r['ci'], index=coef_names)

        def conf_int(alpha=0.05):
            if alpha != 0.05:
                raise NotImplementedError("Only alpha=0.05 is supported, "
                                          "{} passed".format(alpha))
            return ci
        attrs['conf_int'] = conf_int

        return attrs

    def _package_attrs(self, attrs):
        # Sometimes features are retrieved from wrapper (stargazer does this),
        # other times from the actual result (statsmodels' summary_col does
        # this), so we'll have both.
        rres = RRegressionResults()

        # Use patsy to extract the target variable:
        fobj = ModelDesc.from_formula(self.formula)
        rres.target = fobj.lhs_termlist[0].name()
        rres.model = self

        # We need to hijack this rather than subclassing because stargazer does
        # not use "isistance()" but "type()":
        wrap = RegressionResultsWrapper(rres)

        # All items except "params" are @cache_readonly and need first to be
        # deleted, and then redefined:
        for attr in attrs:
            if attr not in 'params':
                if hasattr(rres, attr):
                    delattr(rres, attr)
            setattr(rres, attr, attrs[attr])
            setattr(wrap, attr, attrs[attr])

        rres.__class__ = RegressionResults
        return wrap
