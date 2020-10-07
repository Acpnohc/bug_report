#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 08:31:26 2020

@author: JChonpca_Huang
"""

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import math
import os
import random

os.chdir('data_2')

height = 1
width = 3

people_num = 90
company_num = 9


#model = WorldModel(height, width, people_num, company_num)

global Y
global I
global W

Y = np.zeros([height, width])
I = np.zeros([height, width])
W = np.zeros([height, width])
W = W + 1



delta = 0.6
rho = 0.8
epolish = 5

IW = 1


gamma = 0.5
beta = 0.8
alpha = 0.1
T = 10

def T_tras(i,j,ii,jj):
    
    return T**(math.sqrt((i-ii)**2+(j-jj)**2))

def City_attribute_calus(model, Y, I, W):
    
    YY = Y[:].reshape([model.height*model.width,1])
    II = I[:].reshape([model.height*model.width,1])
    WW = W[:].reshape([model.height*model.width,1])
    
    # updape_p
    
    p = beta/rho*W
    
    global z
    global m
    
    z = np.zeros([model.height, model.width])
    m = np.zeros([model.height, model.width])
    
    for i in range(model.height):
        for j in range(model.width):
            
            tmp_1 = 0
            tmp_2 = 0
            
            cellmates = model.grid.get_cell_list_contents([(i,j)])
            

            for k in cellmates:
                
                if k.class_name == 'company':
                    
                    tmp_1 = tmp_1 + 1
                
                else:
                    
                    tmp_2 = tmp_2 + 1
                
            
            z[i,j] = tmp_1
            m[i,j] = tmp_2

    
    
    
#    print(z)
#    print(m)


    
    def f(x):
        
        #x: YY(model.height*model.width) , II(model.height*model.width) , WW((model.height*model.width))
                
        YY = []
        II = []
        WW = []
        
        for i in range(model.height):
            
            for j in range(model.width):
                
                YY.append(x[i*model.width +j]-10-(m[i,j]*x[2*model.height*model.width+ i*model.width +j]))
                
                tmp_1 = 0
                tmp_2 = 0
                
                for ii in range(model.height):
                    
                    for jj in range(model.width):
                        
                        tmp_1 = tmp_1 + z[ii,jj]*(T_tras(i,j,ii,jj)*p[ii,jj])**(1/(1-epolish))
                        
                        tmp_2 = tmp_2 + (x[ii*model.width +jj]*((x[model.height*model.width+ ii*model.width +jj])**(epolish-1))*((T_tras(i,j,ii,jj))**(1-epolish)))
                
                
                tmp_1 = (tmp_1)**(1/(1-epolish))
                
                tmp_2 = ((tmp_1)**(1/epolish))*(rho)*((beta)**(-rho))*((delta/((epolish-1)*(alpha)))**(1/epolish))
                
                
                II.append(x[model.height*model.width + i*model.width +j]-tmp_1)
                WW.append(x[2*model.height*model.width + i*model.width +j]-tmp_2)
                
                
        inital = np.hstack([YY,II])
        inital = np.hstack([inital,WW])
        
        
        return inital
    
    inital = np.hstack([YY,II])
    inital = np.hstack([inital,WW])
    
 
    result = fsolve(f,inital)
    #print(result)
    return result


     
def City_attribute_calus_inital(model, Y, I, W):
    
    YY = Y[:].reshape([model.height*model.width,1])
    II = I[:].reshape([model.height*model.width,1])

#    WW = W[:]
    
    # updape_p
    
    
    global z
    global m
    
    p = beta/rho*W
    z = np.zeros([model.height, model.width])
    m = np.zeros([model.height, model.width])
    
    for i in range(model.height):
        for j in range(model.width):
            
            tmp_1 = 0
            tmp_2 = 0
            
            cellmates = model.grid.get_cell_list_contents([(i,j)])
            
            for k in cellmates:
                
                if k.class_name == 'company':
                    
                    tmp_1 = tmp_1 + 1
                
                else:
                    
                    tmp_2 = tmp_2 + 1
                
            
            z[i,j] = tmp_1
            m[i,j] = tmp_2
    
#    print(z)
#    print(m)
    
    



    def f(x):
        
        #x: YY(model.height*model.width) , II(model.height*model.width) , WW((model.height*model.width))
                
        YY = []
        II = []
#        WW = []
        
        for i in range(model.height):
            
            for j in range(model.width):

#                YY.append(x[i*model.width +j]-10-(m[i,j]*x[2*model.height*model.width+ i*model.width +j]))
                
                YY.append(x[i*model.width +j]-10-(m[i,j]*1))
                
                tmp_1 = 0
#                tmp_2 = 0
                
                for ii in range(model.height):
                    
                    for jj in range(model.width):
                        
                        tmp_1 = tmp_1 + z[ii,jj]*(T_tras(i,j,ii,jj)*p[ii,jj])**(1/(1-epolish))
                        
#                        tmp_2 = tmp_2 + (x[ii*model.width +jj]*((x[model.height*model.width+ ii*model.width +jj])**(epolish-1))*((T_tras(i,j,ii,jj))**(1-epolish)))
                
                
                tmp_1 = (tmp_1)**(1/(1-epolish))
                
#                tmp_2 = ((tmp_1)**(1/epolish))*(rho)*((beta)**(-rho))*((delta/((epolish-1)*(alpha)))**(1/epolish))
                
                
                II.append(x[model.height*model.width + i*model.width +j]-tmp_1)
#                WW.append(x[2*model.height*model.width + i*model.width +j]-tmp_2)
                
                
        inital = np.hstack([YY,II])
#        YY.extend(WW)
        
        return inital
    
    inital = np.hstack([YY,II])
#    YY.extend(WW)

 
    result = fsolve(f,inital)
    
    return result

    

class Agent(Agent):
    
    def __init__(self, class_name, unique_id, location ,model):
        super().__init__(unique_id, model)
        self.class_name = class_name
        self.people_name = unique_id
        self.location = location
        self.model = model



    def move(self,Y,I,W):

        
        if self.class_name == 'people':
            
            
            tmp_move = np.zeros([self.model.height,self.model.width])
            
            ii = self.location[0]
    
            jj = self.location[1]
            
            for i in range(self.model.height):
                
                for j in range(self.model.width):
    
                    self.model.grid.move_agent(self,(i,j))
                    
                    YY = Y[:]
                    II = I[:]
                    WW = W[:]
                    
                    tmp  = City_attribute_calus(model, YY, II, WW)
            
                    YY = tmp[0:height*width].reshape([height, width])
                    II = tmp[height*width:2*height*width].reshape([height, width])
                    WW = tmp[2*height*width:3*height*width].reshape([height, width])

                    
                    C = delta*W/II
                    
                    total_people = m + 10
            
                    total_food = total_people*1

                    tmp_move[i,j] = ((total_food[i,j])**(1-delta))*((C[i,j])**(delta))
    
                    self.model.grid.move_agent(self,(ii,jj))
            
            tmp_move_ = pd.DataFrame(tmp_move)
            
            


            
            tmp_place = tmp_move_.stack().idxmax()
            print('people_moving')
            print(tmp_move)
            print(tmp_place)
            print(ii,jj)
            
            self.model.grid.move_agent(self, tmp_place)
            
            self.location = list(tmp_place)
            
            self.model.time = self.model.time + 1
            
            tmp = City_attribute_calus(model, Y, I, W)
            print(m)
            print(z)

            Y = tmp[0:height*width].reshape([height, width])
            I = tmp[height*width:2*height*width].reshape([height, width])
            W = tmp[2*height*width:3*height*width].reshape([height, width])
            
            np.savetxt('Y_' + str(model.time) +'.txt',Y)
            np.savetxt('I_' + str(model.time) +'.txt',I)
            np.savetxt('W_' + str(model.time) +'.txt',W)
            np.savetxt('M_' + str(model.time) +'.txt',m)
            np.savetxt('Z_' + str(model.time) +'.txt',z)
        
        elif self.class_name == 'company':
            

            
            tmp_move = np.zeros([self.model.height,self.model.width])
            
            
            for i in range(self.model.height):
                
                for j in range(self.model.width):

                    self.model.grid.move_agent(self,(i,j))

                    YY = Y[:]
                    II = I[:]
                    WW = W[:]
                    
                    tmp  = City_attribute_calus(model, YY, II, WW)
            
                    YY = tmp[0:height*width].reshape([height, width])
                    II = tmp[height*width:2*height*width].reshape([height, width])
                    WW = tmp[2*height*width:3*height*width].reshape([height, width])
                    
                    

                    p = (beta/rho)*W

                    tmp = 0
                    
                    for ii in range(self.model.height):

                        for jj in range(self.model.width):

                            tmp = tmp + (YY[ii,jj]*((II[ii,jj])**(epolish-1))*((T_tras(i,j,ii,jj))**(1-epolish)))

                    Nfr = delta*((beta/rho)**(-epolish))*WW[i,j]*tmp
                                        
                    tmp_move[i,j] = Nfr*p[i,j] - ((alpha + beta*Nfr)*WW[i,j])
                    
                    ii = self.location[0]
                    jj = self.location[1]

                    self.model.grid.move_agent(self,(ii,jj))
                    
            tmp_move = pd.DataFrame(tmp_move)
            
            tmp_place = tmp_move.stack().idxmax()
            
            print('company_moving')
            print(tmp_move)
            print(tmp_place)
            print(ii,jj)
            
            self.model.grid.move_agent(self, tmp_place)
            
            self.location = list(tmp_place)
            
            self.model.time = self.model.time + 1
            
            tmp  = City_attribute_calus(model, Y, I, W)
            print(m)
            print(z)
            Y = tmp[0:height*width].reshape([height, width])
            I = tmp[height*width:2*height*width].reshape([height, width])
            W = tmp[2*height*width:3*height*width].reshape([height, width])
            
            np.savetxt('Y_' + str(model.time) +'.txt',Y)
            np.savetxt('I_' + str(model.time) +'.txt',I)
            np.savetxt('W_' + str(model.time) +'.txt',W)
            np.savetxt('M_' + str(model.time) +'.txt',m)
            np.savetxt('Z_' + str(model.time) +'.txt',z)

            
    def step(self):
        
        self.move(Y,I,W)
#        self.model.env_clean(Y,I,W)
        
#        print(len(self.model.schedule.agents))


class WorldModel(Model):


    def __init__(self, height, width, people_num, company_num):
        
        self.time = 1
        
        self.height = height
        self.width = width
        
        
        self.people_num = people_num
        self.company_num = company_num
        
        #world_map creation
        
        self.grid = MultiGrid(height, width, True)
        self.schedule = RandomActivation(self)
        
        #adding people_agent
        
        for i in range(0,self.people_num):
            
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)

            x = i//((self.people_num/(self.height*self.width))*self.width)
            y = i%((self.people_num/(self.height*self.width))*self.width)//(self.people_num/(self.height*self.width))
            
            
            x = int(x)
            y = int(y)
            
            

            location = [x,y]
            
            a = Agent('people', i, location, self)
            
            self.schedule.add(a)
            self.grid.place_agent(a, (x, y))
            
            
        #adding company_agent
        
        for i in range(self.people_num, self.people_num + self.company_num):
            
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)

            x = (i-self.people_num)//((self.company_num/(self.height*self.width))*self.width)
            y = (i-self.people_num)%((self.company_num/(self.height*self.width))*self.width)//(self.company_num/(self.height*self.width))

            x = int(x)
            y = int(y)


            location = [x,y]
            
            a = Agent('company', i, location, self)
            
            self.schedule.add(a)
            self.grid.place_agent(a, (x, y))


    def env_clean(self,Y,I,W):


        companys = []
        proflits = []

        for i in self.schedule.agents:
            
            if i.class_name == 'company':
                
                

                companys.append(i)
                
                p = beta/rho*W
    
                tmp = 0
                
                k = i.location[0]
                j = i.location[1]
                
                for ii in range(self.height):
    
                    for jj in range(self.width):
    
                        tmp = tmp + (Y[ii,jj]*((I[ii,jj])**(epolish-1))*((T_tras(k,j,ii,jj))**(1-epolish)))
    
                Nfr = delta*((beta/rho)**(-epolish))*W[k,j]*tmp
                                    
                proflit = Nfr*p[k,j] - ((alpha + beta*Nfr)*W[k,j])
                
                
                proflits.append(proflit)
            
            
    
        print(sum(proflits))

        if sum(proflits) > 0:
             

            
            
            print('delete company')

            self.schedule.remove(companys[proflits.index(min(proflits))])

            tmp  = City_attribute_calus(model, Y, I, W)
            
            
            print(m)
            print(z)
            Y = tmp[0:height*width].reshape([height, width])
            I = tmp[height*width:2*height*width].reshape([height, width])
            W = tmp[2*height*width:3*height*width].reshape([height, width])

        
        elif sum(proflits) < 0:
            
            
            print('adding company')
            
            location = companys[proflits.index(max(proflits))].location

            a = Agent('company', -1 , location, self)

            self.schedule.add(a)
            
            self.grid.place_agent(a,tuple(location))
            
            
            tmp  = City_attribute_calus(model, Y, I, W)
            
            print(m)
            print(z)
            Y = tmp[0:height*width].reshape([height, width])
            I = tmp[height*width:2*height*width].reshape([height, width])
            W = tmp[2*height*width:3*height*width].reshape([height, width])

            


    def step(self):
        
        self.schedule.step()

        
        
        
        
        
#height = 10        
#width = 10

#people_num = 1000
#company_num = 100        
        

model = WorldModel(height, width, people_num, company_num)

#Y = np.zeros([height, width])
#I = np.zeros([height, width])
#W = np.zeros([height, width])
#W = W + 1

tmp  = City_attribute_calus_inital(model, Y, I, W)




Y = tmp[0:height*width].reshape([height, width])
I = tmp[height*width:2*height*width].reshape([height, width])

np.savetxt('init_Y.txt',Y)
np.savetxt('init_I.txt',I)
np.savetxt('init_W.txt',W)


for i in range(5):
    
    model.step()
