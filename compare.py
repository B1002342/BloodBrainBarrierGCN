from pymol import cmd
import os
import numpy as np
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

def compute_rmsd(dir, perm):
    derivs = [f for f in os.listdir(dir)]
    nicotine = "pdbs/nicotine.pdb"
    cmd.load(nicotine, "nicotine")
    
    rmsds = []
    
    for deriv in derivs:
        path = os.path.join(dir, deriv)
        if not perm:
            cmd.load(path, f'0_{deriv}')
            rmsd = cmd.align("nicotine", f'0_{deriv}')[0]
        else:
            cmd.load(path, f'1_{deriv}')
            rmsd = cmd.align("nicotine", f'1_{deriv}')[0]
            
        rmsds.append(rmsd)
    
    return rmsds

def check_sig(zero, one):
    print(len(zero), len(one))
    _, shapiro_zero = shapiro(zero)
    _, shapiro_one = shapiro(one)
    if shapiro_zero > 0.05 and shapiro_one > 0.05:
        t, p = ttest_ind(zero, one, equal_var=False)
        print(f"t-test: t = {t:.2f}, p = {p:.2e}")
    else:
        u, p= mannwhitneyu(zero, one, alternative='two-sided')
        print(f"mann-whitney u test: U = {u:.2f}, p = {p:.2e}")
    
def main():
    zeros_dir = "pdbs/0"
    ones_dir = "pdbs/1"
    
    zero_scores = compute_rmsd(zeros_dir, 0)
    one_scores = compute_rmsd(ones_dir, 1)
    
    print("zero: ", np.median(zero_scores))
    print("one: ", np.median(one_scores))
    
    check_sig(zero_scores, one_scores)
    
main()
