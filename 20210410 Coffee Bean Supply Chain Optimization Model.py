#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 08:20:39 2020

@author: rachelriri
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:41:10 2019

@author: rachelriri
"""

import numpy as np
from gurobipy import * # Import Gurobi solver
import pandas as pd
import xlrd
xlrd.xlsx.ensure_elementtree_imported(False, None)
xlrd.xlsx.Element_has_iter = True

# Creating Model

model = Model('Linear Program')

#index=======================================================================================================================
N = 5 # resourcing countriy number
E = 4 # sun,low shade, medium shade, high shade
S = 7 # climate scenario (tem,pre) 
T = 30 # predicte years
Q = 4 # 0-0,0-1,1-0,1-1 fertilization irrigation
L = 10 # ports number
R = 5 # roasting centers number
M = 49 # consuming states

#load parameters data=========================================================================================================

# the unit fixed cost by using shading management from e to e'
F_ec = pd.read_excel('unit shading management cost from e to e prime.xlsx')#$/ha
F_ec = np.asmatrix(F_ec.iloc[:,1:5])


#the unit variable cost by using manamgement practice q
# nn ny ys yy f+i
G_q = pd.read_excel('unit practice management cost.xlsx').values.ravel()#$/ha
#G_q = np.asmatrix(G_q)

# random.random() # the probability of climate scenario s
p_s = pd.read_excel('scenario probability.xlsx').values.ravel()
#p_s = np.asmatrix(p_s)

# the unit ocean shipping cost from source country j to port l 
W_jl = pd.read_excel('the unit ocean shipping cost from source country j to port l _3.xlsx')
W_jl = np.asmatrix(W_jl.iloc[:,1:11])

# the unit transportation cost from port l to roasting center r
J_lr = pd.read_excel('the unit transportation cost from port l to roasting center r_2.xlsx')
J_lr = np.asmatrix(J_lr.iloc[:,1:6])

# the unit transportation cost from roasting center r to state m
K_rm = pd.read_excel('the unit transportation cost from roasting center r to state m_3.xlsx')
K_rm = np.asmatrix(K_rm.iloc[:,1:50])

# the unit cost of roasting in roasting center r 
Q_r = pd.read_excel('unit roasting cost.xlsx').iloc[:,1].values.ravel()
#Q_r = np.asmatrix(Q_r.iloc[:,1])

# the base yield in country j in year t under shading management e
#  
Y_jtes = np.zeros((5,30,4,7))
f_name = 'based_yield_0425'
book = xlrd.open_workbook(f_name+'.xlsx')
sheet = book.sheet_by_index(0)
for j in range(0,N):
    for t in range(0,T):
        for e in range(0,E):
            for s in range(0,S):
                row = 120 * j + 4 * t + e + 1
                col = s + 3
                Y_jtes[j][t][e][s] = sheet.cell(row,col).value

#the base yield in country j in year t under shading management e(e=0)
Y_jt0s = np.zeros((5,30,7))
f_name = 'based_yield_0425_under shade level 0'
book = xlrd.open_workbook(f_name+'.xlsx')
sheet = book.sheet_by_index(0)
for j in range(0,N):
    for t in range(0,T):
        for s in range(0,S):
            row = 30 * j + 1 * t + 1
            col = s + 2
            Y_jt0s[j][t][s] = sheet.cell(row,col).value  
           

# the additional yield in country j in year t under management practice q
B_jtqs = np.zeros((5,30,4,7))
f_name = 'addtional_yield_0425'
book = xlrd.open_workbook(f_name+'.xlsx')
sheet = book.sheet_by_index(0)
for j in range(0,N):
    for t in range(0,T):
        for q in range(0,Q):
            for s in range(0,S):
                row = 120 * j + 4 * t + q + 1
                col = s + 3
                B_jtqs[j][t][q][s] = sheet.cell(row,col).value

# the annual percentage of coffee beans exported
V_j = pd.read_excel('the annual percentage of coffee beans exported.xlsx').iloc[:,1].values.ravel()
#V_j = np.asmatrix(V_j.iloc[:,1])#.ravel()


# the annual percentage of coffee beans exported to US
P_j = pd.read_excel('Percentage of coffee beans exported to U.S.A.xlsx').iloc[:,1].values.ravel()


# roasting centers capacity 
C_rt  = pd.read_excel('roasting center capacity.xlsx')#.values
C_rt  = np.asmatrix(C_rt.iloc[:,1:31])


# coffee beans demand in state m  
D_mt = pd.read_excel('coffee beans demand in 49 states_0425.xlsx',sheet_name='5%')#.values
D_mt = np.asmatrix(D_mt.iloc[:,1:31])

# cultivation area in country j
A_j0 = pd.read_excel('cultivation area at 2021.xlsx').iloc[:,1].values.ravel()#cultivated area in 2020

A_jts = np.zeros((5,30,7))#suitable growing area
f_name = 'cultivation area 0425'
book = xlrd.open_workbook(f_name+'.xlsx')
sheet = book.sheet_by_index(0)
for j in range(0,N):
    for t in range(0,T):
        for s in range(0,S):
            row = 30 * j + t + 1
            col = s + 2
            A_jts[j][t][s] = sheet.cell(row,col).value


#Ψ = min({A_jo,A_jts}
Ψ_jts = np.zeros((5,30,7))#min({A_jo,A_jts}
f_name = 'minimal of A_j0 and A_jts_0425'
book = xlrd.open_workbook(f_name+'.xlsx')
sheet = book.sheet_by_index(0)
for j in range(0,N):
    for t in range(0,T):
        for s in range(0,S):
            row = 30 * j + t + 1
            col = s + 2
            Ψ_jts[j][t][s] = sheet.cell(row,col).value
            
Xi_je0 = pd.read_excel('The area with shade level e in country j before the first stage’s decision-making.xlsx')# the area with shade level e in country j brefore the first stage's decison making
Xi_je0 = np.asmatrix(Xi_je0.iloc[:,1:5])
           
# Creating Variables==========================================================================================================

Rho_jec ={} # the areas that are updated from e to e’ with e<e' in country j
for j in range(0,N):
    for e in range(0,E):
        for c in range(0,E):
            Rho_jec[j,e,c] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="Rho_jec%s,%s,%s"%(j,e,c))

Rho_jce ={} # the areas that are updated from e to e’ with e<e' in country j
for j in range(0,N):
    for c in range(0,E):
        for e in range(0,E):
            Rho_jce[j,c,e] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="Rho_jce%s,%s,%s"%(j,c,e))


Xi_je = {} # the area with shade level e in country j after the first stage's decison making
for j in range(0,N):
    for e in range(0,E):
        Xi_je[j,e] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="Xi_je%s,%s"%(j,e))
        

Sigma = {} # the percentage area that uses shading managment e in country j
for j in range(0,N):
    for e in range(0,E):
        Sigma[j,e] = model.addVar(lb=0.0,ub=1,vtype=GRB.CONTINUOUS,name="Sigma%s,%s"%(j,e))
        

Theta = {} # the percentage area that uses management practice q in country j in year t under climate scenario s
for j in range(0,N):
    for t in range(0,T):
        for q in range(0,Q):
            for s in range(0,S):
                Theta[j,t,q,s] = model.addVar(lb=0.0,ub=1,vtype= GRB.CONTINUOUS,name="Theta%s,%s,%s,%s"%(j,t,q,s))
                
h = {} # the green coffee beans exported from country j to port l
for j in range(0,N):
    for l in range(0,L):
        for t in range(0,T):
            for s in range(0,S):
                h[j,l,t,s] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="h%s,%s,%s,%s"%(j,l,t,s))

u = {} # the green coffee beans transported from port l to rosating center r
for l in range(0,L):
    for r in range(0,R):
        for t in range(0,T):
            for s in range(0,S):
                u[l,r,t,s] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="u%s,%s,%s,%s"%(l,r,t,s))
                
x = {} # the roasted coffee beans transported from roasting center r to state m
for r in range(0,R):
    for m in range(0,M):
        for t in range(0,T):
            for s in range(0,S):
                x[r,m,t,s] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="x%s,%s,%s,%s"%(r,m,t,s))

O = {} # shortage in year t
for m in range(0,M):
    for t in range(0,T):
        for s in range(0,S):
            O[m,t,s]= model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="O%s,%s,%s"%(m,t,s))
                
U_ts = {} #total transportation cost
for t in range(0,T):
    for s in range(0,S):
        U_ts[t,s] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="U_ts%s,%s"%(t,s))

H_rts = {} # roasting total cost
for r in range(0,R):
    for t in range(0,T):
        for s in range(0,S):
            H_rts[r,t,s] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="H_rts%s,%s,%s"%(r,t,s))
                      


# Creating Objective Founction================================================================================================
# first stage cost + second stage cost + roasting cost + transportation cost

model.setObjective(quicksum(F_ec[e,c] * Rho_jec[j,e,c] for j in range (0,N) for e in range(0,E) for c in range(0,E) if c!=e)
                 + quicksum(p_s[s] * G_q[q] * Theta[j,t,q,s] * A_jts[j,t,s] for j in range(0,N) for t in range(0,T) for s in range(0,S) for q in range(0,Q))
                 + quicksum(p_s[s] * U_ts[t,s] for t in range(0,T) for s in range(0,S))
                 + quicksum(O[m,t,s] * 10000 for m in range(0,M) for t in range(0,T) for s in range(0,S))
                 + quicksum(u[l,r,t,s] * Q_r[r] for l in range(0,L) for t in range (0,T) for s in range(0,S)),GRB.MINIMIZE)




# Adding Constraints==========================================================================================================
# yiekld*loss >= exported  
for t in range(0,T):
    for s in range(0,S):
        for j in range(0,N):
              model.addConstr(quicksum(Y_jtes[j,t,e,s] * Sigma[j,e] * Ψ_jts[j,t,s] * 0.79 * 0.4 * P_j[j] * V_j[j] for e in range(0,E)) 
                              +  0.79 * 0.4 * P_j[j] * V_j[j] * Y_jt0s[j,t,s] * (A_jts[j,t,s]- Ψ_jts[j,t,s])
                        + quicksum(B_jtqs[j,t,q,s] * Theta[j,t,q,s] * A_jts[j,t,s] * 0.79 * 0.4 * P_j[j] * V_j[j] for q in range(0,Q)) 
                        >= quicksum(h[j,l,t,s] for l in range(0,L))) 
              
# sum(j to l to r to m) = U
for t in range(0,T):
    for s in range(0,S):
        model.addConstr(quicksum(h[j,l,t,s] * W_jl[j,l] for j in range(0,N) for l in range(0,L))
                      + quicksum(u[l,r,t,s] * J_lr[l,r] for l in range(0,L) for r in range(0,R))
                      + quicksum(x[r,m,t,s] * K_rm[r,m] for r in range(0,R) for m in range(0,M))
                      == U_ts[t,s])

#the area  hectare (ha) with shade level e in country j before the first stage’s decision-making 
#substract the areas that are updated from e to e’ with e<e' in country j
#add the areas that are updated from e' to e with e<e' in country j
#equal the area  hectare (ha) with shade level e in country j after the first stage’s decision-making 
for e in range(0,E):
    for j in range(0,N):
        model.addConstr(Xi_je0[j,e] - quicksum(Rho_jec[j,e,c] for c in range(0,E) if c!=e) 
                        + quicksum(Rho_jce[j,c,e] for c in range(0,E) if c!=e) == Xi_je[j,e])

# =============================================================================
# for e in range(0,E):
#     for j in range(0,N):
#         model.addConstr(quicksum(Xi_je0[j,e] for e in range(0,E) ) == quicksum(Xi_je[j,e] for e in range(0,E)))
# 
# =============================================================================

for j in range(0,N):
    model.addConstr(Sigma[j,e] * A_j0[j] == Xi_je[j,e])

for j in range(0,N):
    model.addConstr(quicksum(Xi_je[j,e] for e in range(0,E))==A_j0[j])
    

###########################################################
                                              
# exported from country j to port l * loss >= transported from port l to roasting center r
for t in range(0,T):
    for s in range(0,S):
        for l in range(0,L):
            model.addConstr(quicksum(h[j,l,t,s] for j in range(0,N)) >= quicksum(u[l,r,t,s] for r in range(0,R)))
             
# transported from port l to roasting center r * loss >= transported from roasting center r to state m         
for t in range(0,T):
    for s in range(0,S):
        for r in range(0,R):
            model.addConstr(0.79 * quicksum(u[l,r,t,s] for l in range(0,L)) >= quicksum(x[r,m,t,s] for m in range (0,M)))
            

# transported from port l to roasting center r <= roasting center capacity 
for t in range(0,T):
    for s in range(0,S):
        for r in range(0,R):
            model.addConstr(quicksum(u[l,r,t,s] for l in range(0,L)) <= C_rt[r,t])
            
# transported from roasting center r to state m  >= demand            
for t in range(0,T):
    for s in range(0,S):
        for m in range(0,M):
            model.addConstr(quicksum(x[r,m,t,s] for r in range(0,R)) + O[m,t,s] >= D_mt[m,t])

# roasting cost
for t in range(0,T):
    for s in range(0,S):
        for r in range(0,R):
            model.addConstr(quicksum(u[l,r,t,s] * Q_r[r] for l in range(0,L)) == H_rts[r,t,s])
            
# the total percentage area that uses shading managment e in country j = 1   
for j in range(0,N):
    model.addConstr(quicksum(Sigma[j,e] for e in range(0,E)) == 1)

# the total percentage area that uses  management practice q in country j in year t under climate scenario s  
for t in range (0,T):
    for s in range(0,S):
        for j in range(0,N):
            model.addConstr(quicksum(Theta[j,t,q,s] for q in range(0,Q)) == 1)

# =============================================================================
# if model.solCount == 0:
#     print("Model is infeasible")
#     model.computeIIS()
#     model.write("model_iis.ilp")
# =============================================================================

  
model.write('model.lp') 
model.optimize()             
          
# Print solution==========================================================================================================
print("======objective value =======")
obj = model.getObjective()
print(obj.getValue())
   
print("======Sigma =======")
for j in range(0,N):
    for e in range(0,E):
        var_name = model.getVarByName("Sigma%s,%s"%(j,e))
        print("Sigma%s,%s"%(j,e), "%g"%(var_name.X))

print("====== Theta ======")
for j in range(0,N):
    for t in range(0,T):
        for q in range(0,Q):
            for s in range(0,S):
                var_name = model.getVarByName("Theta%s,%s,%s,%s"%(j,t,q,s))
                print("Theta%s,%s,%s,%s"%(j,t,q,s), "%g"%(var_name.X))
        

# print("====== h ======")
# for j in range(0,N):
#     for l in range(0,L):
#         for t in range(0,T):
#             for s in range(0,S):
#                 var_name = model.getVarByName("h%s,%s,%s,%s"%(j,l,t,s))
#                 print("h%s,%s,%s,%s"%(j,l,t,s),"%g"%(var_name.X))                

# print("====== u ======")
# for l in range(0,L):
#     for r in range(0,R):
#         for t in range(0,T):
#             for s in range(0,S):
#                 var_name = model.getVarByName("u%s,%s,%s,%s"%(l,r,t,s))
#                 print("u%s,%s,%s,%s"%(l,r,t,s),"%g"%(var_name.X))
                
# print("====== x ======")
# for r in range(0,R):
#     for m in range(0,M):
#         for t in range(0,T):
#             for s in range(0,S):
#                 var_name = model.getVarByName("x%s,%s,%s,%s"%(r,m,t,s))
#                 print("x%s,%s,%s,%s"%(r,m,t,s),"%g"%(var_name.X))
                
# print("====== U_ts =====")
# for t in range(0,T):
#     for s in range(0,S):
#         var_name = model.getVarByName("U_ts%s,%s"%(t,s))
#         print("U_ts%s,%s"%(t,s), "%g"%(var_name.X)) 
                
# print("====== H_rts ======") 
# for r in range(0,R):
#     for t in range(0,T):
#         for s in range(0,S):
#             var_name = model.getVarByName("H_rts%s,%s,%s"%(r,t,s))
#             print("H_rts%s,%s,%s"%(r,t,s), "%g"%(var_name.X)) 

model.write("04_25_mysolution test data_5%.sol")










