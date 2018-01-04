# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:36:34 2017

@author: karandesai
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.autograd as Variable

#import data
movies=pd.read_csv("ml-1m/movies.dat",sep="::",header=None,engine='python',encoding="latin")
ratings=pd.read_csv("ml-1m/ratings.dat",sep="::",header=None,engine='python',encoding="latin")
users=pd.read_csv("ml-1m/users.dat",sep="::",header=None,engine='python',encoding="latin")

train=pd.read_csv("ml-100k/u1.base",delimiter='\t')
test=pd.read_csv("ml-100k/u1.test",delimiter='\t')

train=np.array(train,dtype='int')
test=np.array(test,dtype='int')

nb_users=int(max(max(train[:,0]),max(test[:,0])))
nb_movies=int(max(max(train[:,1]),max(test[:,1])))

def convert(data):
    new_data=[]
    for id_user in range(1,nb_users+1):
        id_movies=data[:,1][data[:,0]==id_user]
        id_ratings=data[:,2][data[:,0]==id_user]
        ratings=np.zeros(nb_movies)
        ratings[id_movies-1]=id_ratings
        new_data.append(list(ratings))
    return new_data

train=convert(train)
test=convert(test)

#torchh tensors
train=torch.FloatTensor(train)
test=torch.FloatTensor(test)
train[train==0]=-1
train[train==1]=0
train[train==2]=0
train[train>=3]=1
test[test==0]=-1
test[test==1]=0
test[test==2]=0
test[test>=3]=1


#class RBM
class RBM():
    def __init__(self,nh,nv):
        self.w=torch.randn(nh,nv)
        self.a=torch.randn(1,nh)
        self.b=torch.randn(1,nv)
    def sample_h(self,x):
        wx=torch.mm(x,self.w.t())
        activation=wx+self.a.expand_as(wx)
        ph_given_v=torch.sigmoid(activation)
        return ph_given_v,torch.bernoulli(ph_given_v)
    def sample_v(self,y):
        wy=torch.mm(y,self.w)
        activation=wy+self.b.expand_as(wy)
        pv_given_h=torch.sigmoid(activation)
        return pv_given_h,torch.bernoulli(pv_given_h)
    def train(self,v0,p0,vk,pk):
        self.w+=torch.mm(v0.t(),p0)-torch.mm(vk.t(),pk)
        self.b+=torch.sum((v0-vk),0)
        self.a+=torch.sum((p0-pk),0)
batch=100    
nv=nb_movies
nh=100
rbm=RBM(nh,nv)
            
#training the RBM
nb_epochs=10
for epoch in range(1,nb_epochs+1):
    train_loss=0.
    s=0.
    for id_user in range(0,nb_users-batch,batch):
        v0=train[id_user:id_user+batch]
        vk=train[id_user:id_user+batch]
        p0,_=rbm.sample_h(v0)
        for k in range(10):
            _,hk=rbm.sample_h(vk)
            _,vk=rbm.sample_v(hk)
            vk[v0==-1]=v0[v0==-1]
        pk,_=rbm.sample_h(vk)
        rbm.train(v0,p0,vk,pk)
        train_loss+=torch.mean(torch.abs(v0[v0>=0]-vk[vk>=0]))
        s+=1.
    print('epoch:',epoch,'loss:',(train_loss/s))
            


test_loss=0.
s=0.
for id_user in range(nb_users):
    v=train[id_user:id_user+1]
    vt=test[id_user:id_user+1]
    if len(vt[vt>=0])>0:
        _,h=rbm.sample_h(v)
        _,v=rbm.sample_v(h)
        
        test_loss+=torch.mean(torch.abs(v[vt>=0]-vt[vt>=0]))
        s+=1.
print('loss:',(test_loss/s))
