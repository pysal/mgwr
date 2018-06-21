import numpy as np
from spglm.family import Gaussian, Binomial, Poisson
from spglm.glm import GLM


def summaryModel(self):
    summary = '=' * 75 + '\n'
    summary += "%-54s %20s\n" % ('Model type', self.family.__class__.__name__)
    summary += "%-60s %14d\n" % ('Number of observations:', self.n)
    summary += "%-60s %14d\n\n" % ('Number of covariates:', self.k)
    return summary

def summaryGLM(self):
    
    XNames = ["X"+str(i) for i in range(self.k)]
    glm_rslt = GLM(self.model.y,self.model.X,constant=False,family=self.family).fit()

    summary = "%s\n" %('Global Regression Results')
    summary += '-' * 75 + '\n'
    summary += "%-62s %12.3f\n" %  ('Residual sum of squares:', glm_rslt.deviance)
    summary += "%-62s %12.3f\n" %  ('Log-likelihood:', glm_rslt.llf)
    summary += "%-62s %12.3f\n" %  ('AIC:', glm_rslt.aic)
    #summary += "%-62s %12.3f\n" %  ('AICc:', glm_rslt.aicc)
    summary += "%-62s %12.3f\n" %  ('BIC:', glm_rslt.bic)
    summary += "%-62s %12.3f\n" %  ('R2:', glm_rslt.D2)
    summary += "%-62s %12.3f\n\n" % ('Adj. R2:', glm_rslt.adj_D2)
    
    summary += "%-31s %10s %10s %10s %10s\n" % ('Variable', 'Est.', 'SE' ,'t(Est/SE)', 'p-value')
    summary += "%-31s %10s %10s %10s %10s\n" % ('-'*31, '-'*10 ,'-'*10, '-'*10,'-'*10)
    for i in range(self.k):
        summary += "%-31s %10.3f %10.3f %10.3f %10.3f\n" % (XNames[i], glm_rslt.params[i], glm_rslt.bse[i], glm_rslt.tvalues[i], glm_rslt.pvalues[i])
    summary += "\n"
    return summary

def summaryGWR(self):
    XNames = ["X"+str(i) for i in range(self.k)]
    
    summary = "%s\n" %('Geographically Weighted Regression (GWR) Results')
    summary += '-' * 75 + '\n'

    if self.model.fixed:
        summary += "%-50s %20s\n" % ('Spatial kernel:', 'Fixed ' + self.model.kernel)
    else:
        summary += "%-54s %20s\n" % ('Spatial kernel:', 'Adaptive ' + self.model.kernel)

    summary += "%-62s %12.3f\n" % ('Bandwidth used:', self.model.bw)
    summary += "%-62s %12.3f\n" % ('Residual sum of squares:', self.resid_ss)
    summary += "%-62s %12.3f\n" % ('Effective number of parameters (trace(S)):', self.tr_S)
    summary += "%-62s %12.3f\n" % ('Residual Degree of freedom (n - trace(S)):', self.df_model)
    summary += "%-62s %12.3f\n" % ('Sigma estimate:', np.sqrt(self.sigma2))
    summary += "%-62s %12.3f\n" % ('Log-likelihood:', self.llf)
    summary += "%-62s %12.3f\n" % ('AIC:', self.aic)
    summary += "%-62s %12.3f\n" % ('AICc:', self.aicc)
    summary += "%-62s %12.3f\n" % ('BIC:', self.bic)

    if isinstance(self.family, Gaussian):
        #summary += "%-60s %12.6f\n" % ('CV:', 1.0)
        summary += "%-62s %12.3f\n" % ('R2:', self.R2)
        #summary += "%-60s %12.6f\n" % ('Adj. R2:', self.adjR2)

        #summary += "%-60s %12.6f\n" % ('Null deviance:', 0)
        #summary += "%-60s %12.6f\n" % ('Residual deviance:', 0)
        #summary += "%-60s %12.6f\n" % ('Percent deviance explained:', 0)

    summary += "%-62s %12.3f\n" % ('Adj. alpha at 95%:', self.adj_alpha[1])
    #summary += "%-60s %12.6f\n" % ('Adj. t-value at 95%:', self.critical_tval(self.adj_alpha[1]))

    summary += "\n"

    summary += "%s\n" % ('Summary Statistics For Varying (Local) Parameter Estimates')
    summary += '-' * 75 + '\n'
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('Variable', 'Mean' ,'STD', 'Min' ,'Median', 'Max')
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('-'*20, '-'*10 ,'-'*10, '-'*10 ,'-'*10, '-'*10)
    for i in range(self.k):
        summary += "%-20s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % (XNames[i], np.mean(self.params[:,i]) ,np.std(self.params[:,i]),np.min(self.params[:,i]) ,np.median(self.params[:,i]), np.max(self.params[:,i]))

    summary += '=' * 75 + '\n'

    return summary



def summaryMGWR(self):
    summary = ''
    summary += "%s\n" %('Multi-Scale Geographically Weighted Regression (MGWR) Results')
    
    summary += '-' * 75 + '\n'
    
    if self.model.fixed:
        summary += "%-50s %20s\n" % ('Spatial kernel:', 'Fixed ' + self.model.kernel)
    else:
        summary += "%-54s %20s\n" % ('Spatial kernel:', 'Adaptive ' + self.model.kernel)

    summary += "%s\n" % ('Model settings')
    summary += '-' * 75 + '\n'

    summary += "%-45s %s\n" % ('Criterion for optimal bandwidth:', self.optimCriDropdown.currentText())
    summary += "%-45s %s\n" % ('Initialization choice:', self.initBeta)
    summary += "%-45s %s\n" % ('Score of Change (SOC) type:', self.SOC)
    summary += "%-45s %s\n\n" % ('Termination criterion for MGWR:', self.tol_multi)

    summary += "\n%s\n" %('MGWR bandwidth selection')
    summary += '-' * 75 + '\n'
    summary += "%-20s %30s %20s\n" % ('Variable', 'Optimal Bandwidth', 'ENP')
    for j in range(len(self.XNames)):
        summary += "%-20s %30.3f %20.3f\n" % (self.XNames[j], self.bws[j], self.results.ENP_j[j])

    summary += "\n%s\n" % ('Diagnostic information')
    summary += '-' * 75 + '\n'
    summary += "%-60s %12.6f\n" % ('Residual sum of squares:', self.resid_ss)
    summary += "%-60s %12.6f\n" % ('Effective number of parameters (trace(S)):', self.tr_S)
    summary += "%-60s %12.6f\n" % ('Residual Degree of freedom (n - trace(S)):', self.df_model)
    summary += "%-60s %12.6f\n" % ('Sigma estimate:', np.sqrt(self.sigma2))
    summary += "%-60s %12.6f\n" % ('-2Log-likelihood:', -2*self.llf)
    summary += "%-60s %12.6f\n" % ('Classic AIC:', self.aic)
    summary += "%-60s %12.6f\n" % ('AICc:', self.aicc)
    summary += "%-60s %12.6f\n" % ('BIC:', self.bic)
    
    
    summary += "%s\n" % ('Summary Statistics For Varying (Local) Parameter Estimates')
    summary += '-' * 75 + '\n'
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('Variable', 'Mean' ,'STD', 'Min' ,'Median', 'Max')
    summary += "%-20s %10s %10s %10s %10s %10s\n" % ('-'*20, '-'*10 ,'-'*10, '-'*10 ,'-'*10, '-'*10)
    for i in range(self.k):
        summary += "%-20s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % (XNames[i], np.mean(self.params[:,i]) ,np.std(self.params[:,i]),np.min(self.params[:,i]) ,np.median(self.params[:,i]), np.max(self.params[:,i]))

    summary += '=' * 75 + '\n'
    return summary

















