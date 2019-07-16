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
		self.mgwr_bw = []

		self.gwr_aicc = []
		self.gwr_bic=[]
		'''
		self.gwr_bic = []
		self.gwr_aicc = []
		self.gwr_predy = []
		'''
		self.gwr_params = []
		self.mgwr_aicc = []
		'''
		self.mgwr_bic = []
		self.mgwr_aicc = []
		self.mgwr_predy = []
		'''
		self.mgwr_params = []
		self.mgwr_bic=[]

def add(a,b):
	return 1+((1/120)*(a+b))

def con(u,v):
	return (0*(u)*(v))+0.3

def sp(u,v):
	return 1+1/3240*(36-(6-u/2)**2)*(36-(6-v/2)**2)

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
	error = np.random.normal(0,0.1,625)

	B0=con(X,Y).reshape(-1,1)
	B1=add(X,Y).reshape(-1,1)
	B2=sp(X,Y).reshape(-1,1)

	lat=Y.reshape(-1,1)
	lon=X.reshape(-1,1)

	param = np.hstack([B0,B1,B2])
	cons=np.ones_like(x1)
	X=np.hstack([cons,x1,x2])
	y=(np.exp((np.sum(X * param, axis=1)+error).reshape(-1, 1)))
	y_new = np.random.poisson(y)

	coords = np.array(list(zip(lon,lat)))
	y = np.array(y_new).reshape((-1,1))
	X1=np.hstack([x1,x2])

	bw=Sel_BW(coords,y,X1,family=Poisson(),offset=None)
	bw=bw.search()
	s.gwr_bw.append(bw)
	gwr_model=GWR(coords,y,X1,bw,family=Poisson(),offset=None).fit()
	s.gwr_aicc.append(gwr_model.aicc)
	s.gwr_bic.append(gwr_model.bic)
	s.gwr_params.append(gwr_model.params)
	selector=Sel_BW(coords,y,X1,multi=True,family=Poisson(),offset=None)
	selector.search()
	s.mgwr_bw.append(selector.bw[0])
	mgwr_model=MGWR(coords,y,X1,selector,family=Poisson(),offset=None).fit()
	s.mgwr_aicc.append(mgwr_model.aicc)
	s.mgwr_bic.append(mgwr_model.bic)
	s.mgwr_params.append(mgwr_model.params)
	output.put(s)
	return
