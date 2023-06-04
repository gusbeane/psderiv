import arepo
import numpy as np
from joblib import Parallel,delayed
import sys
from tqdm import tqdm
import glob

from util import generate_bins, Fourier_analysis, identify_bar_region, compute_pattern_speed

def get_pos_vel_acc_mass(sn):
    center = sn.part1.pos.value[np.argmin(sn.part1.pot.value)]
    
    pos = []
    vel = []
    acc = []
    mass = []
    
    for pt in [2, 3, 4]:
        if hasattr(sn, 'part'+str(i)):
            part = getattr(sn, 'part'+str(i))
            pos.append(part.pos.value - center)
            vel.append(part.vel.value)
            acc.append(part.acce)
            
            if sn.MassTable[pt]>0:
                mass.append(np.full(sn.NumPart_Total[pt], sn.MassTable[pt]))
            else:
                mass.append(part.mass.value)
    
    pos = np.concatenate(pos)
    vel = np.concatenate(vel)
    acc = np.concatenate(acc)
    mass = np.concatenate(mass)
    

def _runner(path, ic, name, snap, ptypes=[1, 2, 3, 4]):
    sn = arepo.Snapshot(path + '/output/', snap, 
                        parttype=ptypes, 
                        fields=['Coordinates', 'Velocities', 'Acceleration', 'Masses', 'Potential'],
                        combineFiles=True)
    
    center = sn.part1.pos.value[np.argmin(sn.part1.pot.value)]
    pos = sn.part2.pos.value - center
    vel = sn.part2.vel.value
    acc = sn.part2.acce
    mass = np.full(sn.NumPart_Total[2], sn.MassTable[2])
    
    bins = generate_bins(pos)
    R, A2, phi2 = Fourier_analysis(pos, mass, bins)
    is_barred, R0, R1, A2max, phi2_mean = identify_bar_region(bins, A2, phi2)
    ps, psdot = compute_pattern_speed(pos, vel, acc, mass, R0, R1)
    
    time = sn.Time.value

    # Package it all together
    output = (time, is_barred, R0, R1, A2max, phi2_mean, ps, psdot)
    
    return output

def run(path, ic, name, nsnap, nproc):

    out = Parallel(n_jobs=nproc) (delayed(_runner)(path, ic, name, i) for i in tqdm(range(nsnap)))

    time      = np.array([out[i][0] for i in range(len(out))])
    is_barred = np.array([out[i][1] for i in range(len(out))])
    R0        = np.array([out[i][2] for i in range(len(out))])
    R1        = np.array([out[i][3] for i in range(len(out))])
    A2max     = np.array([out[i][4] for i in range(len(out))])
    phi2_mean = np.array([out[i][5] for i in range(len(out))])
    ps        = np.array([out[i][6] for i in range(len(out))])
    psdot     = np.array([out[i][7] for i in range(len(out))])

    out = {'time'      : time,
           'is_barred' : is_barred,
           'R0'        : R0,
           'R1'        : R1,
           'A2max'     : A2max,
           'phi2_mean' : phi2_mean,
           'ps'        : ps,
           'psdot'     : psdot}
    
    np.save('ps_'+name+'.npy', out)

if __name__ == '__main__':
    nproc = int(sys.argv[1])

    basepath = '/n/holylfs05/LABS/hernquist_lab/Users/abeane/psderiv/iso/'

    Nbody = 'Nbody'
    SMUGGLE = 'SMUGGLE'

    pair_list = [(Nbody, 'lvl4'), # 0
                 (Nbody, 'lvl3'), # 1
                 (SMUGGLE, 'lvl4'), # 2
                 (SMUGGLE, 'lvl3'), # 3
                 ]

    name_list = [           p[0] + '-' + p[1] for p in pair_list]
    path_list = [basepath + 'runs/' + p[0] + '/' + p[1] for p in pair_list]
    ic_list   = [basepath + 'ics/' + p[0] + '/' + p[1] for p in pair_list]
    
    nsnap_list = [len(glob.glob(path+'/output/snapdir*/*.0.hdf5')) for path in path_list]
  
    i = int(sys.argv[2])
    path = path_list[i]
    name = name_list[i]
    nsnap = nsnap_list[i]
    ic = ic_list[i]

    out = run(path, ic, name, nsnap, nproc)
