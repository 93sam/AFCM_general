import cv2
import numpy as np 
from reader import ReadImage

np.seterr(divide='ignore', invalid='ignore')

class AFCM:
	def __init__(self,imageName,n_clusters,epsilon=0.05,max_iter=-1):
		self.m = 2
		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.epsilon = epsilon

		read = ReadImage(imageName)
		self.X, self.numPixels = read.getData()
		self.X = self.X.astype(np.float)
		print self.X

		self.U = []
		for i in range(self.numPixels):
			index = i % n_clusters
			l = [ 0 for j in range(n_clusters) ]
			l[index] = 1
			self.U.append(l)
		self.U = np.array(self.U).astype(np.float)
		print self.U,self.U.shape

		self.C = []
		l = 0
		for i in range(n_clusters):
			self.C.append(l)
		self.C = np.array(self.C).astype(np.float)
		print self.C, self.C.shape

		self.G = []
		l = 1
		for i in range(self.numPixels):
			self.G.append(l)
		self.G = np.array(self.G).astype(np.float)
		print self.G, self.G.shape

	def eucledian_dist(self,a,b):
		return np.linalg.norm(a-b)

	def compute_U(self):
		for i in range(self.numPixels):
			for j in range(self.n_clusters):
				numer = self.eucledian_dist(self.X[i],self.G[i]*self.C[j]) ** (-2/(self.m - 1))
				denom = 0
				for k in range(self.n_clusters):
					denom += self.eucledian_dist(self.X[i],self.G[i]*self.C[k]) ** (-2/(self.m - 1))
				self.U[i][j] = numer/denom
				#print numer,denom,'\n'
				print self.eucledian_dist(self.X[i],self.G[i]*self.C[j]),self.eucledian_dist(self.X[i],self.G[i]*self.C[j])**-2, '\n'
		'''		
		for i in range(self.numPixels):
			for j in range(self.n_clusters):
				print self.U[i][j]
		'''

def main():
	cluster = AFCM('MRI.jpg',3)
	cluster.compute_U()

if __name__ == '__main__':
	main()