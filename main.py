from ModalAnalysis import ModalAnalysis as ma
from BMO import BMO as bmo_algo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    best_solutions=[]
    save_file='convergence_plots_2D_100_50_distributed'
    for i in range(10):
        file_name = '2D-data.xlsx'
        dimension = int(file_name[0])
        elements = pd.read_excel(file_name, sheet_name='Sheet1').values[:, 1:]
        nodes = pd.read_excel(file_name, sheet_name='Sheet2').values[:, 1:]
        arrested_dofs=np.array([0,1,42,43])
        # arrested_dofs=np.arange(0,24)
        aa = ma(elements, nodes, dimension,arrested_dofs=arrested_dofs)
        M=aa.assembleMass()

        x_exp=np.zeros(len(elements))
        x_exp[5]=0.35
        x_exp[23]=0.20
        x_exp[15]=0.4   #localized
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
        # print(objective_function(np.zeros(shape=x_exp.shape)))

        optimizer = bmo_algo(n_vars=len(elements),cost_fn=objective_function,verbose=True,max_generations=200,population_size=200,guess=0.1)
        
        log=optimizer.run()

        plt.yscale('log')
        plt.plot(np.abs(log))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title(f'Convergence plot {i}')
        plt.savefig(f'./{save_file}/convergence_{i}.png')
        plt.clf()

        best_solutions.append(optimizer.best_.reshape(1,-1))
        print('*'*80)
    
    best_solutions=np.concatenate(best_solutions,axis=0)
    np.savetxt(f'./{save_file}/data.csv',best_solutions,delimiter=',')


if __name__=='__main__':
    main()