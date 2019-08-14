import sys
sys.path.append("C:/Users/msachde1/Downloads/Research/Development/mgwr")
import pandas as pd
import numpy as np

from mgwr.gwr import GWR
from spglm.family import Gaussian, Binomial, Poisson
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW

class stats(object):
	def __init__(self):
		self.gwr_bw = []
		self.gwr_aicc = []
		self.gwr_bic=[]
		self.gwr_aic=[]
		self.gwr_params = []
		self.gwr_predy = []
		self.gwr_rss = []

		self.mgwr_bw = []
		self.mgwr_aicc = []
		self.mgwr_bic=[]
		self.mgwr_aic=[]
		self.mgwr_params = []
		self.mgwr_predy = []
		self.mgwr_rss = []

def add(a,b):
	return 1+((1/12)*(a+b))

def con(u,v):
	return (0*(u)*(v))+0.3

def sp(u,v):
	return 1+1/324*(36-(6-u/2)**2)*(36-(6-v/2)**2)

def med(u,v):
    B = np.zeros((25,25))
    for i in range(25):
        for j in range(25):

            if u[i][j]<=8:
                B[i][j]=0.2
            elif u[i][j]>17:
                B[i][j]=0.7
            else:
                B[i][j]=0.5
    return B

class foo(object):
	def __init__(self):
		self.x = []
		self.name = ""
		self.num = 0

def fkj(name, output):
	ff = foo()
	ff.x = [1]
	ff.name = name
	output.put(ff)
	return

def models(output):
	print("start of the function")
	s = stats()
	x = np.linspace(0, 25, 25)
	y = np.linspace(25, 0, 25)
	X, Y = np.meshgrid(x, y)
	x1=np.random.normal(0,1,625).reshape(-1,1)
	x2=np.random.normal(0,1,625).reshape(-1,1)
	#x3=np.random.normal(0,1,625).reshape(-1,1)
	error = np.random.normal(0,0.1,625)

	B0=con(X,Y).reshape(-1,1)
	#B1=add(X,Y).reshape(-1,1)
	B2=sp(X,Y).reshape(-1,1)
	B3=med(X,Y).reshape(-1,1)

	lat=Y.reshape(-1,1)
	lon=X.reshape(-1,1)

	param = np.hstack([B0,B2,B3])
	cons=np.ones_like(x1)
	X=np.hstack([cons,x1,x2])
	y_raw = ((np.exp(np.sum(X * param, axis=1)+error)/(1+(np.exp(np.sum(X * param, axis=1)+error))))).reshape(-1,1)
	#y_raw=(np.exp((np.sum(X * param, axis=1)+error).reshape(-1, 1)))
	#y_new = np.random.poisson(y_raw)
	y_new = np.random.binomial(1,y_raw,(625,1))

	coords = np.array(list(zip(lon,lat)))
	y = np.array(y_new).reshape((-1,1))
	X1=np.hstack([x1,x2])

	bw=Sel_BW(coords,y,X1,family=Binomial())
	bw=bw.search()
	s.gwr_bw.append(bw)
	gwr_model=GWR(coords,y,X1,bw,family=Binomial()).fit()
	s.gwr_aicc.append(gwr_model.aicc)
	s.gwr_bic.append(gwr_model.bic)
	s.gwr_aic.append(gwr_model.aic)
	s.gwr_params.append(gwr_model.params)
	s.gwr_predy.append(gwr_model.predy)
	s.gwr_rss.append(gwr_model.resid_ss)

	selector=Sel_BW(coords,y,X1,multi=True,family=Binomial())
	selector.search()
	s.mgwr_bw.append(selector.bw[0])
	mgwr_model=MGWR(coords,y,X1,selector,family=Binomial()).fit()
	s.mgwr_aicc.append(mgwr_model.aicc)
	s.mgwr_bic.append(mgwr_model.bic)
	s.mgwr_aic.append(mgwr_model.aic)
	s.mgwr_params.append(mgwr_model.params)
	s.mgwr_predy.append(mgwr_model.predy)
	s.mgwr_rss.append(mgwr_model.resid_ss)

	return s
