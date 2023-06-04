import numpy as np
from window import window, grad_window, grad2_window

def generate_bins(pos,
                  Nmin=4000,
                  Nmax=32000,
                  DexBin=0.1):
    R = np.linalg.norm(pos[:,:2], axis=1)
    R = R[np.argsort(R)]
    
    bins = []
    
    Rmin = 0
    i = Nmin - 1
    Rmax = R[i]
    bins.append([Rmin, Rmax])
    
    # create first set of bins
    
    while Rmax < 10:
        Rmin = R[i+1]
        
        Rmax_ = 10.**(DexBin) * Rmin
        key_Rmax = np.where(R < Rmax_)[0]
        Rmax = R[key_Rmax[-1]]
        N = len(np.where(np.logical_and(R >= Rmin, R <= Rmax))[0])
        
        if N < Nmin:
            Rmax = R[i+Nmin]
            N = Nmin
        elif N > Nmax:
            Rmax = R[i+Nmax]
            N = Nmax
        
        bins.append([Rmin, Rmax])
        i = i + N - 1
    
    bins_with_int = []
    
    # second interleaving set of bins
    for i in range(len(bins)-1):
        bin0 = bins[i]
        bin1 = bins[i+1]
        
        Rmin = (bin0[0] + bin0[1])/2.
        Rmax = (bin1[0] + bin1[1])/2.
    
        bins_with_int.append(bin0)
        bins_with_int.append([Rmin, Rmax])
    
    bins_with_int.append(bin1)
    
    return bins_with_int

def Fourier_analysis(pos, mass, bins):
    R = np.linalg.norm(pos[:,:2], axis=1)
    
    cphi = pos[:,0]/R
    sphi = pos[:,1]/R
    c2phi = (pos[:,0]**2 - pos[:,1]**2)/R**2
    s2phi = 2*pos[:,0]*pos[:,1]/R**2
    
    Sigma2_bins = []
    Sigma0_bins = []
    phi2_bins = []
    R_bins = []
    
    for Rmin, Rmax in bins:
        key = np.logical_and(R >= Rmin, R <= Rmax)
        C2 = np.sum(mass[key] * c2phi[key])
        S2 = np.sum(mass[key] * s2phi[key])
        
        Sigma2 = np.sqrt(C2**2 + S2**2)
        Sigma0 = np.sum(mass[key])
        phi2 = 0.5 * np.arctan2(S2, C2)
        
        R_bins.append(np.mean(R[key]))
        
        Sigma2_bins.append(Sigma2)
        Sigma0_bins.append(Sigma0)
        phi2_bins.append(phi2)
    
    R_bins = np.array(R_bins)
    Sigma2_bins = np.array(Sigma2_bins)
    Sigma0_bins = np.array(Sigma0_bins)
    phi2_bins = np.array(phi2_bins)
    
    return R_bins, Sigma2_bins/Sigma0_bins, phi2_bins

def identify_bar_region(bins, A2, phi2,
                        bar_A2_thresh=0.2, Dphi2_thresh=10*np.pi/180.,
                        min_A2_thresh=0.05):
    
    A2max = np.max(A2)

    if A2max < bar_A2_thresh:
        is_barred = False
        return is_barred, np.nan, np.nan, A2max, np.nan
    else:
        is_barred = True
    
    lower_i = np.argmax(A2)
    upper_i = lower_i
    
    phi2_list = [phi2[lower_i]]
    Dphi2 = 0.0
    phi2_mean = phi2[lower_i]
    
    while True:
        lower_i_ = lower_i - 1
        upper_i_ = upper_i + 1
        
        lower_eligible = A2[lower_i_] > A2max/2.
        upper_eligible = A2[upper_i_] > A2max/2.
        
        phi2_mean = np.mean(phi2_list)
        
        if lower_eligible and upper_eligible:
            phi2_lower = phi2[lower_i_]
            phi2_upper = phi2[upper_i_]
            
            if np.abs(phi2_upper - phi2_mean) < np.abs(phi2_lower - phi2_mean):
                phi2_list.append(phi2_upper)
                Dphi2 = np.max(phi2_list) - np.min(phi2_list)
                if Dphi2 < Dphi2_thresh:
                    upper_i = upper_i_
                    continue
                else:
                    break
            else:
                phi2_list.append(phi2_lower)
                Dphi2 = np.max(phi2_list) - np.min(phi2_list)
                if Dphi2 < Dphi2_thresh:
                    lower_i = lower_i_
                    continue
                else:
                    break
        
        if lower_eligible and not upper_eligible:
            phi2_list.append(phi2_lower)
            Dphi2 = np.max(phi2_list) - np.min(phi2_list)
            if Dphi2 < Dphi2_thresh:
                lower_i = lower_i_
                continue
            else:
                break
                
        if upper_eligible and not lower_eligible:
            phi2_list.append(phi2_upper)
            Dphi2 = np.max(phi2_list) - np.min(phi2_list)
            if Dphi2 < Dphi2_thresh:
                upper_i = upper_i_
                continue
            else:
                break
        
        if not upper_eligible and not lower_eligible:
            break
    
    Rmin = bins[lower_i][0]
    Rmax = bins[upper_i][1]
    
    return is_barred, Rmin, Rmax, A2max, phi2_mean

def compute_pattern_speed(pos, vel, acc, mass, R0, R1):
    # compute R
    R = np.linalg.norm(pos[:,:2], axis=1)
    
    # compute cos and sin terms
    cphi = pos[:,0]/R
    sphi = pos[:,1]/R
    c2phi = (pos[:,0]**2 - pos[:,1]**2)/R**2
    s2phi = 2*pos[:,0]*pos[:,1]/R**2
    
    # compute vel and acc in cylindrical coordinates
    velR = vel[:,0] * cphi + vel[:,1] * sphi
    velphi = - vel[:,0] * sphi + vel[:,1] * cphi
    
    accR = acc[:,0] * cphi + acc[:,1] * sphi
    accphi = - acc[:,0] * sphi + acc[:,1] * cphi
    
    # compute window functions
    W = window(R, R0, R1)
    dW = grad_window(R, R0, R1)
    ddW = grad2_window(R, R0, R1)
    
    # compute C2, S2, Cdot2, and Sdot2
    C2 = np.sum(mass * W * c2phi)
    S2 = np.sum(mass * W * s2phi)
    
    Cdot2 = np.sum(mass * (velR*dW*c2phi - 2 * (velphi/R) * W * s2phi))
    Sdot2 = np.sum(mass * (velR*dW*s2phi + 2 * (velphi/R) * W * c2phi))
    
    #compute Cddot2, Sddot2
    term1 = 2*(velphi/R)*velR*dW + W*(accphi/R)
    term1 *= -2*mass*s2phi
    term2 = -4*W*(velphi/R)**2 + dW*accR + velR**2*ddW
    term2 *= mass*c2phi
    Cddot2 = np.sum(term1 + term2)
    
    term1 = 2*(velphi/R)*velR*dW + W*(accphi/R)
    term1 *= 2*mass*c2phi
    term2 = -4*W*(velphi/R)**2 + dW*accR + velR**2*ddW
    term2 *= mass*s2phi
    Sddot2 = np.sum(term1 + term2)
    
    # compute pattern speed from C, S terms
    Omegap = 0.5 * (C2*Sdot2 - S2*Cdot2) / (C2**2 + S2**2)
    
    # compute time derivative of pattern speed
    term1 = 2*(S2*Cdot2 - C2*Sdot2)*(C2*Cdot2 + S2*Sdot2)
    term2 = (C2**2 + S2**2)*(-S2*Cddot2 + C2*Sddot2)
    Omegapdot = (term1 + term2)/(2*(C2**2+S2**2)**2)
    
    return Omegap, Omegapdot
