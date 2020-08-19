import numpy as np
import pandas as pd
from rpy2.robjects import globalenv, r, pandas2ri
from rpy2.rinterface import embedded
pandas2ri.activate()

from statsmodels.regression.linear_model import (RegressionResults,
                                                 RegressionResultsWrapper)

from statsmodels.api import OLS

from scipy import stats
from patsy import ModelDesc

from .r_translate import _df_from_r
from .fake_number import FakeNumber

class RRegressionResults:
    """
    This holds results from a model estimated with R, but mimicks the API of
    one estimated with statsmodels.
    """
    def __init__(self):
        # statsmodels.regression.linearmodel.RegressionResults.__init__()
        # initializese the cache, which is otherwise managed at the class
        # level. So let's do it here.
        self._cache = {}

        # All actual work instead is done directly on instances.

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

        if not 'terms' in d_res:
            msg = ("Interpreting r objects inside Python is only supported "
                   "for few estimators. More will work using "
                   "RModel.from_rdata() directly.")
            raise NotImplementedError(msg)

        formula = str(d_res['terms']).splitlines()[0]

        # We want to create a fake dataset, and we use patsy to get the list of
        # variables. We are actually creating columns for interactions and
        # functions too... but who cares, identifying them would be at the
        # moment overkill.
        fobj = ModelDesc.from_formula(formula)
        varnames = [t.name()
                    for t in fobj.rhs_termlist + fobj.lhs_termlist][1:]

        # We need to pass some pd.DataFrame to from_formula() below - but it
        # doesn't seem to be actually used.
        data = pd.DataFrame(-1, index=[0], columns=[0])

        # Creating the OLS object and only then hijacking it allows us to best
        # profit of statsmodels' machinery:
        mod = OLS.from_formula(formula, data)
        mod.__class__ = RModel

        attrs = mod._inspect_R(rsum, ci=ci)
        wrap = mod._package_attrs(attrs)

        return wrap

    @classmethod
    def from_rda(cls, filename, objname):
        """
        Load the summary of an R model from a .rda file as statsmodels-like
        estimation result.
        Such file can be created with the "save()" command in R.

        Parameters
        ----------
        filename : str
            Path of the file to load from.
        objname : str
            Name of the object to load from the file.

        Examples
        --------
        If an object named "regsum" was saved in R with the command
            save(regsum, file = "/home/pietro/r_results.rda")
        then it can be reloaded by calling this command as
            res = RModel.from_rda("/home/pietro/r_results.rda", "regsum")
        """

        r['load'](filename)

        d_res = cls._r_as_dict(None, r[objname])
        try:
            ci = r("ci <- confint({})".format(objname))
        except embedded.RRuntimeError:
            ci = None

        # FIXME: while this works differently from the code building the
        # coefficients matrix in _inspect_R (which does not retrieve from R),
        # there is clearly room for de-duplication.
        coefs = cls._get_coeffs_mat(None, objname)

        items = list(coefs.index)

        try:
            # E.g. mfx marginal effects
            target = str(d_res['call'][1][1])
            formula = " ~ ".join([target, " + ".join(items)])
            columns = [target] + items
        except IndexError:
            # E.g. OLS
            # This is ugly...
            formula = str(d_res['terms']).splitlines()[0]
            target = formula.split(' ')[0].split('~')[0]

        data = pd.DataFrame(-1, index=[0], columns=[target]+items)

        # Creating the OLS object and only then hijacking it allows us to best
        # profit of statsmodels' machinery:
        mod = OLS.from_formula(formula, data)
        mod.__class__ = RModel

        attrs = mod._inspect_R(objname)
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

        attrs = self._inspect_R('rsum')

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

    def _get_coeffs_mat(self, rsumname, d_res=None):
        if d_res is None:
            # Awkward, but allows self to be None:
            d_res = RModel._r_as_dict(None, r[rsumname])

        if "coefficients" in d_res:
            return _df_from_r("{}$coefficients".format(rsumname))
        elif 'mfxest' in d_res:
            return _df_from_r("{}$mfxest".format(rsumname))

        raise NotImplementedError("Could not recognize estimation model.")

    def _inspect_R(self, rsumname, rresname=None):
        """
        Extract from an R estimation summary the various pieces.

        Parameters
        ----------
        rsumname : string
            Name of estimation summary in the R environment.
        rresname : string, default None
            Name of estimation results in the R environment, if present.
            Typically required in order to retrieve confidence intervals.
        """

        res = r[rsumname]
        d_res = self._r_as_dict(res)

        coeffs_mat = self._get_coeffs_mat(rsumname, d_res)
        coef_names = coeffs_mat.index

        # R denotes the intercept as "(Intercept)", statsmodels as "Intercept":
        intercept_idces = np.where(coef_names == '(Intercept)')[0]
        if len(intercept_idces):
            # An index is immutable, so transform to np.array:
            coef_names = coef_names.values
            coef_names[intercept_idces[0]] = 'Intercept'

        # Retrieve main results:
        # FIXME: get by label, not index
        attrs = {
        'params' : pd.Series(coeffs_mat.iloc[:,0], index=coef_names),
        'tvalues' : pd.Series(coeffs_mat.iloc[:,2], index=coef_names),
        'pvalues' : pd.Series(coeffs_mat.iloc[:,3], index=coef_names),
        'bse' : pd.Series(coeffs_mat.iloc[:,1], index=coef_names),
        'rsquared' : d_res.get('r.squared',
                               [FakeNumber("No R^2 available")])[0],
        'rsquared_adj' : d_res.get('adj.r.squared',
                               [FakeNumber("No adjusted R^2 available")])[0],
        'scale' : (d_res['sigma'][0]**2 if 'sigma' in d_res else
                   FakeNumber("No sigma/scale available"))
        }

        f_exc = FakeNumber("No f statistics available")
        types = {'fvalue' : float, 'df_model' : int, 'df_resid' : int}
        if 'fstatistic' in d_res:
            # E.g. OLS
            for attr, idx in zip(['fvalue', 'df_model', 'df_resid'], range(3)):
                attrs[attr] = types[attr](d_res['fstatistic'][idx])
        elif 'fit' in d_res:
            # E.g. mfx marginal effects:
            attrs['fvalue'] = f_exc
            dfit = self._r_as_dict(d_res['fit'])
            for attr, label in [('df_model', 'df.null'),
                                ('df_resid', 'df.residual')]:
                attrs[attr] = int(dfit[label])
        else:
            for attr in 'fvalue', 'df_model', 'df_resid':
                attrs[attr] = f_exc

        # Couldn't find this ready in the R summary of an OLS:
        if 'fstatistic' in d_res:
            attrs['f_pvalue'] = stats.f.sf(attrs['fvalue'],
                                           attrs['df_model'],
                                           attrs['df_resid'])
        else:
            attrs['f_pvalue'] = FakeNumber("No f statistics available")

        if rresname is not None:
            try:
                # Retrieve confidence intervals:
                ci = r("ci <- confint({})".format(rresname))
                ci = pd.DataFrame(r['ci'], index=coef_names)
            except embedded.RRuntimeError:
                ci = None
        if rresname is None or ci is None:
            msg = ("Trying to access the confidence intervals of a RModel "
                   "which wasn't passed any.")
            ci = pd.DataFrame(FakeNumber(msg),
                              index=coef_names, columns=range(2))

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
