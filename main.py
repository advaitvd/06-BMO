from ModalAnalysis import ModalAnalysis as ma
from BMO import BMO as bmo_algo
import numpy as np
import pandas as pd

def main():
    file_name = '2D-data.xlsx'
    dimension = int(file_name[0])
    elements = pd.read_excel(file_name, sheet_name='Sheet1').values[:, 1:]
    nodes = pd.read_excel(file_name, sheet_name='Sheet2').values[:, 1:]
    aa = ma(elements, nodes, dimension)
    M=aa.assembleMass()

    x_exp=np.zeros(len(elements))
    x_exp[5]=0.35
    x_exp[23]=0.20
    x_exp[15]=0.4
    x_exp[10]=0.24

    K=aa.assembleStiffness(x_exp)
    w_exp, v_exp=aa.solve_eig(K,aa.M)
    
    num_modes=10

    w_exp=w_exp[:num_modes]
    v_exp=v_exp[:,:num_modes]
    F_exp=np.sum(v_exp*v_exp,axis=0)/(w_exp*w_exp)
    # print("w_exp",w_exp)

    def objective_function(x):
        K=aa.assembleStiffness(x)
        w, v = aa.solve_eig(K, aa.M)
        w=w[:num_modes]
        v=v[:,:num_modes]
        # print(w.shape,v.shape)
        # print('w',w)
        
        MAC=(np.sum((v*v_exp),axis=0)**2)/(np.sum(v*v,axis=0)*np.sum(v_exp*v_exp,axis=0))
        
        F=np.sum(v*v,axis=0)/(w*w)
        MACF=(np.sum(F*F_exp)**2)/(np.sum(F*F)*np.sum(F_exp*F_exp))

        MDLAC=(np.abs(w-w_exp)/w_exp)**2

        # print('MAC, MDLAC',MAC, MDLAC)

        cost = np.sum(1-MAC)+np.sum(MDLAC)+np.sum(1-MACF)
        return cost
    
    print(objective_function(x_exp))

    optimizer = bmo_algo(n_vars=len(elements),cost_fn=objective_function,verbose=True,max_generations=200, guess=0)
    
    optimizer.run()
    optimizer.plot_graph()

    print(optimizer.best_fitness_,optimizer.best_)

if __name__=='__main__':
    main()