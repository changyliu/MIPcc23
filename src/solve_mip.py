import numpy as np
import pandas as pd
import time
import random
import math
from collections import defaultdict
import math
import os
import json

from pyscipopt import Model

def read_data(data_path):

    model = Model("mipcc23")
    model.readProblem(data_path)

    return model

def solve_mip(model, warm_start, instance_name, soln_dir):
    startTime = time.time()
    instance_sol = ''

    if warm_start:
        print('ADD WARMSTART')

        # soln_path = os.path.join(soln_dir, instance_name+'_best.sol')
        # cur_soln = model.readSolFile(soln_path)
        # model.addSol(cur_soln)

        instances = []

        for instance in os.listdir(soln_dir):
            if instance.endswith('.sol'):
                instances.append(instance)
        instances.sort()

        for instance_sol in instances:
            soln_path = os.path.join(soln_dir, instance_sol)
            # read solution
            cur_soln = model.readSolFile(soln_path)
            # print(type(cur_soln))
            # print(model.checkSol(cur_soln))
            # check validity of solution
            if instance_sol[0:10] == instance_name[0:10]:
                print('no feasible warmstart solution found')
                break

            if model.checkSol(cur_soln):
                print(instance_name)
                print('add to solution sets: ', instance_sol)
                model.addSol(cur_soln)
                # break
                # for i, (key, value) in enumerate(saved_ws_soln.items()):
                #     model.setSolVal(ws_sol, vars[i], value)
    
    # print(model.getSols())

    soln_check_time = time.time() - startTime

    solve_start_time = time.time()

    model.optimize()

    tot_solve_time = time.time() - solve_start_time

    if model.getNSols() > 0:
        sol = model.getBestSol()

    else:
        print("No solution found")

    solve_status = model.getStatus()
    obj_value = model.getObjVal()
    tot_node = model.getNTotalNodes()
    presolve_time = model.getPresolvingTime()
    solve_time = model.getSolvingTime()
    num_runs = 0
    primal_bound_soln = model.getNSolsFound()
    
    return solve_status, tot_solve_time, soln_check_time, presolve_time, solve_time, obj_value, sol, tot_node, num_runs, primal_bound_soln, instance_sol


if __name__ == "__main__":
    warm_start = True
    dataset = 'vary_obj/series_1'
    # dataset = 'vary_rhs/series_2'
    datapath = os.path.join('/Users/chang/PhD_workplace/MIPcc23/datasets', dataset)
    soln_dir = os.path.join('/Users/chang/PhD_workplace/MIPcc23/solutions', dataset)
    instances = []
    for instance in os.listdir(datapath):
        if instance.endswith('.gz'):
            instances.append(instance)
    instances.sort()
    # print(instances)

    output_df = pd.DataFrame()
    for instance in instances:
        instance_name = instance.split('.')[0]
        problempath = os.path.join(datapath, instance)
        print(instance)
        model = read_data(problempath)
        model.setParam('display/verblevel',1)
        # model.hideOutput()
        start_time = time.time()
        solve_status, tot_solve_time, soln_check_time, presolve_time, solve_time, obj_value, opt_sol, tot_node, num_runs, primal_bound_soln, instance_sol = solve_mip(model, warm_start, instance_name, soln_dir)
        tot_time = time.time() - start_time

        print(instance)
        print('solve status: ', solve_status)
        print('solve time: ', solve_time)
        print('obj value: ', obj_value)
        print('tot num nodes:, ', tot_node)
        print('num_runs: ', num_runs)
        print('primal_bound_soln: ', primal_bound_soln)

        output_dict = {'instance':      instance_name,
                       'solve_status':  solve_status,
                       'tot_solve_time':    tot_solve_time,
                       'soln_check_time': soln_check_time,
                       'tot_time':      tot_time,
                       'presolve_time': presolve_time,
                       'solve_time':    tot_solve_time,
                       'obj_value':     obj_value,
                       'tot_node':      tot_node,
                       'num_runs':      num_runs,
                       'num_primal_bound_soln': primal_bound_soln,
                       'warmstart_instance':    instance_sol}
        
        output_df = output_df.append(output_dict, ignore_index=True)

        instance_sol_path = os.path.join(soln_dir, '{}_best.sol'.format(instance_name))
        stats_path = os.path.join('/Users/chang/PhD_workplace/MIPcc23/stats/', dataset, '{}_ws{}_previous_stats.txt'.format(instance_name, warm_start))
        model.writeBestSol(instance_sol_path)
        model.writeStatistics(stats_path)

        # print(type(opt_sol))
        # print(type(model.printBestSol(opt_sol)))
        # # print(model.printBestSol(opt_sol))

        # with open(instance+'_soln.json', 'w') as file:
        #     file.write(json.dumps(opt_sol))
            
        # print('opt soln: ', opt_sol)
    
    csv_path = os.path.join('/Users/chang/PhD_workplace/MIPcc23/results/', '{}_ws{}.csv'.format(dataset, warm_start))

    # output_df.to_csv('/Users/chang/PhD_workplace/MIPcc23/results/vary_rhs_series2_no_ws.csv')
    # output_df.to_csv('/Users/chang/PhD_workplace/MIPcc23/results/vary_rhs_series2_ws_own.csv')
    # output_df.to_csv('/Users/chang/PhD_workplace/MIPcc23/results/vary_rhs_series2_ws_previous.csv')
    # output_df.to_csv('/Users/chang/PhD_workplace/MIPcc23/results/vary_obj_series1_ws_previous.csv')
    # output_df.to_csv('/Users/chang/PhD_workplace/MIPcc23/results/vary_obj_series1_ws_own.csv')
    output_df.to_csv('/Users/chang/PhD_workplace/MIPcc23/results/vary_obj_series1_ws_previous.csv')