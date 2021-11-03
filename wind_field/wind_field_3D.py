# -*- coding: utf-8 -*-
"""
Updated 11-2019

@author: bernardc
"""

import numpy as np
import math

def wind_field_3D_func(node_coor_wind, V, Ai, Cij, I, iLj, T, sample_freq, spectrum_type):
    """
    Return the 3 components of the wind speed at each node. shape:(4,num_nodes,num_time_points). U,u,v,w respectively, with U=V+u \n
    Return non-dimensional auto-spectra. shape:(3,num_nodes,num_freq) \n
    Return co-spectra. shape:(3,3,num_freq,num_nodes,num_nodes) \n
    \n
    node_coord_wind -- All coordinates are in the wind flow axes Xf(along wind),Yf,Zf(vertical) in meters. shape:(num_nodes,3)\n
    V -- mean wind speeds at each node in m/s.  shape:(num_nodes) \n
    A -- Auto-spectrum coefficients [Au, Av, Aw] \n
    Cij -- Co-spectrum decay coefficients [Cux,Cuy,Cuz,Cvx,Cvy,Cvz,Cwx,Cwy,Cwz] \n
    I -- Turbulence intensity. [[node1_Iu, node1_Iv, node1_Iw],[node2_Iu..]..]. shape:(num_nodes,3) \n
    iLj -- Integral length scales [xLu,yLu,zLu,xLv,yLv,zLv,xLw,yLw,zLw]. shape:(9, num_nodes) \n
    T -- wind duration in sec. \n
    sample_freq -- number of time points per second, in hertz. \n
    spectrum_type -- =1 as in Y.L.Xu[1] or =2 as adapted from Davenport and E.Cheynet[2] \n
    (compare coherence and cross-correlation between g_nodes, for both types) \n
    \n
    [1] https://www.sciencedirect.com/science/article/pii/S0022460X04001373 \n
    [2] https://se.mathworks.com/matlabcentral/fileexchange/50041-wind-field-simulation \n
    [3] reference from "Theory of Bridge Aerodynamics - Einar Strommen. 2nd ed."
    """
    num_nodes = len(node_coor_wind)
    
    nodes_x = node_coor_wind[:,0]  # in wind flow coordinates! (along flow)
    nodes_y = node_coor_wind[:,1]  # in wind flow coordinates! (horizontal-across flow)
    nodes_z = node_coor_wind[:,2]  # in wind flow coordinates! (usually-vertical flow)
    
    Au,Av,Aw = Ai
    
    Cux,Cuy,Cuz,Cvx,Cvy,Cvz,Cwx,Cwy,Cwz = Cij
    
    xLu,yLu,zLu,xLv,yLv,zLv,xLw,yLw,zLw = iLj
    
    series_N = int(T * sample_freq + 1) # Total number of time points
    series_t = np.linspace(0, T, series_N) # Array of time points
    
    # standard deviations of wind speed for each point
    sigmau = I[:,0] * V
    sigmav = I[:,1] * V
    sigmaw = I[:,2] * V
    sigma = np.moveaxis(np.array([I[:,0]*V , I[:,1]*V , I[:,2]*V]) , -1, 0) 
    
    # array of frequencies used in the spectra
    freq = np.arange(1/T, sample_freq/2, 1/T)
    num_freq = len(freq)

    # --------------
    # Alternative non-equidistant frequency array
    # (more freq points in the lower frequencies than in the higher)
#    freq_low_lim = 1/T # Hz
#    freq_high_lim = sample_freq/2 # Hz
#    freq=[]
#    num_freq = 2000
#    for n in list(range(num_freq)): 
#        freq.append(freq_low_lim+n**2*(freq_high_lim-freq_low_lim)/num_freq**2) # the "2" can be changed
#    freq = np.array(freq)
     # --------------
    
    # delta_freq is the width of each frequency band.
    delta_freq = np.zeros(num_freq)
    delta_freq[0] = (freq[1]-freq[0])/2
    delta_freq[1:-1] = (freq[2:]-freq[1:-1])/2 + (freq[1:-1]-freq[0:-2])/2
    delta_freq[-1] = (freq[-1]-freq[-2])/2
    
    # auto-spectrum (One point Spectrum). N400 eq.(5.5) and eq.(5.4). In f [Hz]
    n_hat = np.zeros([num_nodes, 3, num_freq])    # 3 components, u, v, w
    n_hat[:,0,:] = np.einsum('f,n->nf', freq, xLu/V)
    n_hat[:,1,:] = np.einsum('f,n->nf', freq, xLv/V)
    n_hat[:,2,:] = np.einsum('f,n->nf', freq, xLw/V)
    autospec = np.zeros([num_nodes, 3, num_freq]) # 3 components, u, v, w
    var1 = np.einsum('nf,n->nf', (Au*n_hat[:,0,:])/(1+1.5*Au*n_hat[:,0,:])**(5/3), sigmau**2)
    autospec[:,0,:] = np.einsum('nf,f->nf',var1,1/freq)
    var1 = None
    var2 = np.einsum('nf,n->nf', (Av*n_hat[:,1,:])/(1+1.5*Av*n_hat[:,1,:])**(5/3), sigmav**2)
    autospec[:,1,:] = np.einsum('nf,f->nf',var2,1/freq)
    var2 = None
    var3 = np.einsum('nf,n->nf', (Aw*n_hat[:,2,:])/(1+1.5*Aw*n_hat[:,2,:])**(5/3), sigmaw**2)
    autospec[:,2,:] = np.einsum('nf,f->nf',var3,1/freq)
    var3 = None
    var4 = np.einsum('nif,f->nif', autospec, freq)
    autospec_nondim = np.moveaxis(np.einsum('nif,ni->nif', var4, 1/sigma**2), 1, 0)
    var4 = None

    # Re-shaping AutoSpec from (Nodes*3*Freq) to (Nodes*3*3*Freq), with 0's outside the [3*3] diagonal
    autospec_dummy = np.zeros((num_nodes,3,3,num_freq))
    autospec_dummy[:,0,0,:] = autospec[:,0,:]
    autospec_dummy[:,1,1,:] = autospec[:,1,:]
    autospec_dummy[:,2,2,:] = autospec[:,2,:]
    autospec = autospec_dummy

    # The following is according to the paper "Buffeting response of long-span cable-supported bridges under skew winds - L.D. Zhu, Y.L. Xu"
    # https://www.sciencedirect.com/science/article/pii/S0022460X04001385
    # U_bar(z) and xLu,xLv,xLw are assumed to be the average at the two concerning g_nodes
    
    # eq.(33)
    V_avg   = ( V[:,np.newaxis]  +  V ) /2 # [n * m] matrix
    xLu_avg = (xLu[:,np.newaxis] + xLu) /2 # [n * m] matrix
    xLv_avg = (xLv[:,np.newaxis] + xLv) /2 # [n * m] matrix
    xLw_avg = (xLw[:,np.newaxis] + xLw) /2 # [n * m] matrix

    delta_x = np.abs(nodes_x[:, np.newaxis] - nodes_x)  # [n * m] matrix
    delta_y = np.abs(nodes_y[:, np.newaxis] - nodes_y)  # [n * m] matrix
    delta_z = np.abs(nodes_z[:, np.newaxis] - nodes_z)  # [n * m] matrix

    # Co-spectrum from L.D.Zhu paper and thesis.
    if spectrum_type == 1:  # todo: are we sure this was developed in radians, even though it has f (hertz) as input?
        nxu = np.einsum('mnf,mn->mnf', math.gamma(5/6)/(2*np.sqrt(np.pi)*math.gamma(1/3))*np.sqrt(1+70.78*np.einsum('mnf,mn->mnf',np.einsum('f,mn->mnf',freq, xLu_avg),1/V_avg)**2) , V_avg/xLu_avg )
        # nxu_approx_confirm = np.sqrt( np.tile(freq,(num_nodes,1))**2 + 1/70.78 * (np.transpose(np.tile(V/xLu,(num_freq,1)))**2)) # confirmation
        nxv = np.einsum('mnf,mn->mnf', math.gamma(5/6)/(2*np.sqrt(np.pi)*math.gamma(1/3))*np.sqrt(1+70.78*np.einsum('mnf,mn->mnf',np.einsum('f,mn->mnf',freq, xLv_avg),1/V_avg)**2) , V_avg/xLv_avg )
        nxw = np.einsum('mnf,mn->mnf', math.gamma(5/6)/(2*np.sqrt(np.pi)*math.gamma(1/3))*np.sqrt(1+70.78*np.einsum('mnf,mn->mnf',np.einsum('f,mn->mnf',freq, xLw_avg),1/V_avg)**2) , V_avg/xLw_avg )
        # eq.(32b)
        f_hat_u = np.einsum('mnw,mn->mnw', nxu , np.divide(np.sqrt( (Cux*delta_x)**2 + (Cuy*delta_y)**2 + (Cuz*delta_z)**2 )  ,  V_avg )) # not "2*nxu..." because ".../ V_avg"
        f_hat_v = np.einsum('mnw,mn->mnw', nxv , np.divide(np.sqrt( (Cvx*delta_x)**2 + (Cvy*delta_y)**2 + (Cvz*delta_z)**2 )  ,  V_avg ))
        f_hat_w = np.einsum('mnw,mn->mnw', nxw , np.divide(np.sqrt( (Cwx*delta_x)**2 + (Cwy*delta_y)**2 + (Cwz*delta_z)**2 )  ,  V_avg ))
        f_hat = np.zeros((num_nodes,num_nodes,3,3,num_freq)) # one could just define the diagonals and leave 0's elsewhere, given that later cross-spectruns Sa1a2 between components are 0, for a1 != a2.
        f_hat_dummy = np.array([f_hat_u, f_hat_v, f_hat_w])
        for i in range(3):
            for j in range(3):
                f_hat[:,:,i,j,:] = (f_hat_dummy[i]+f_hat_dummy[j])/2
        # eq.(31)
        R_aa = (1-f_hat)*np.e**(-f_hat)
        # eq.(30). This might give negative Coherences...
        S_aa = np.einsum('nmijf,nmijf->nmijf' , np.sqrt( np.einsum('nijf,mijf->nmijf' , autospec, autospec )) , R_aa, optimize=True )
        # Re-shaping S_aa from [g_nodes*g_nodes*3*3*freq] to [3*3*freq*g_nodes*g_nodes]
        S_aa_dummy = np.moveaxis(S_aa, -1, 0)
        S_aa_dummy = np.moveaxis(S_aa_dummy, -1, 0)
        S_aa_reshaped_radians = np.moveaxis(S_aa_dummy, -1, 0)  # results in radians, according to L.D.Zhu!
        # Finally, in Hertz:
        S_aa_reshaped = S_aa_reshaped_radians * 2*np.pi  # not intuitive! S(f)*delta_f = S(w)*delta_w. eq(2.75) Strommen

    # Alternative co-spectrum, similar to that used by Etienne [2]. Developed in Hertz.
    if spectrum_type == 2:
        f_hat_u_alternative = np.einsum('f,mn->mnf', freq , np.sqrt( (Cux*delta_x)**2 + (Cuy*delta_y)**2 + (Cuz*delta_z)**2 )  /  V_avg )
        f_hat_v_alternative = np.einsum('f,mn->mnf', freq , np.sqrt( (Cvx*delta_x)**2 + (Cvy*delta_y)**2 + (Cvz*delta_z)**2 )  /  V_avg )
        f_hat_w_alternative = np.einsum('f,mn->mnf', freq , np.sqrt( (Cwx*delta_x)**2 + (Cwy*delta_y)**2 + (Cwz*delta_z)**2 )  /  V_avg )
        f_hat_alternative = np.zeros((num_nodes,num_nodes,3,3,num_freq)) # to save time one can just define the diagonals and leave 0's elsewhere, given that later cross-spectruns Sa1a2 between components are 0, for a1 != a2.
        f_hat_alternative[:,:,0,0,:] = f_hat_u_alternative
        f_hat_alternative[:,:,1,1,:] = f_hat_v_alternative
        f_hat_alternative[:,:,2,2,:] = f_hat_w_alternative
        R_aa_alternative = np.e**(-f_hat_alternative)
        S_aa_alternative = np.einsum('nmijf,nmijf->nmijf' , np.sqrt( np.einsum('nijf,mijf->nmijf' , autospec, autospec )) , R_aa_alternative, optimize=True)
        # Re-shaping S_aa from [g_nodes*g_nodes*3*3*freq] to [3*3*freq*g_nodes*g_nodes]
        S_aa_alternative_dummy = np.moveaxis(S_aa_alternative, -1, 0)
        S_aa_alternative_dummy = np.moveaxis(S_aa_alternative_dummy, -1, 0)
        S_aa_alternative_reshaped = np.moveaxis(S_aa_alternative_dummy, -1, 0)
        S_aa_reshaped = S_aa_alternative_reshaped
    
    # Cholesky
    Gmn_u = np.linalg.cholesky(S_aa_reshaped[0,0,:,:,:])
    Gmn_v = np.linalg.cholesky(S_aa_reshaped[1,1,:,:,:])
    Gmn_w = np.linalg.cholesky(S_aa_reshaped[2,2,:,:,:])
    
    # Random Phase angles. [3] pg. 274
    PSI_u = np.random.uniform(low=0, high=2*np.pi, size=(num_nodes,num_freq))
    PSI_v = np.random.uniform(low=0, high=2*np.pi, size=(num_nodes,num_freq))
    PSI_w = np.random.uniform(low=0, high=2*np.pi, size=(num_nodes,num_freq))
        
    # =============================================================================
    # Calculating Time Series U, u, v and w: 
    # =============================================================================

    ## SLOW VERSION (easier to understand) - See [3] eq.(A.16):
    ## Time Series "series_U" 
    #series_U = np.zeros([num_nodes, series_N])
    #var10 = 0
    #for t in list(range(series_N)):
    #    for m in list(range(num_nodes)):
    #        for n in list(range(num_nodes)):
    #            if (n <= m):
    #                for f in list(range(num_freq)):
    #                    var10 += np.abs(Gmn_u[f,m,n])*np.sqrt(2*delta_freq[f])*np.cos(freq[f]*2*np.pi*series_t[t]+PSI[n,f])
    #        series_U[m,t] = var10 + V[m]
    #        var10 = 0
    #var10=None
     
    # FAST VERSION (needs memory) - See [3] eq.(A.16):
    # Time Series "series_u" 
    var12 = np.repeat(PSI_u[:,:,np.newaxis], series_N, axis=2) # PSI_u[n, f, t]
    var13 = np.repeat(freq[:,np.newaxis], series_N, axis=1) # freq[f, t]
    var14 = np.einsum('ft,t->ft', var13 , 2*np.pi*series_t) # omega*t[f,t]
    var13 = None
    var15 = np.repeat(var14[np.newaxis,:,:], num_nodes, axis=0) # omega*t[n,f,t]
    var14 = None
    var16 = np.einsum('f,nft->nft', np.sqrt(2*delta_freq), np.cos(var15+var12), optimize=True)
    var12 = None
    var15 = None
    var17 = np.repeat(V[:,np.newaxis], series_N, axis=1) # V[m,t]
    series_U = np.einsum('fmn,nft->mt', np.abs(Gmn_u), var16, optimize=True) + var17 # U = u + V
    var16 = None
    var17 = None

    series_u = series_U - np.repeat(V[:,np.newaxis], series_N, axis=1) # u = U - V

    # Time Series "series_v"
    var12 = np.repeat(PSI_v[:,:,np.newaxis], series_N, axis=2) # PSI_v[n, f, t]
    var13 = np.repeat(freq[:,np.newaxis], series_N, axis=1) # freq[f,t]
    var14 = np.einsum('ft,t->ft', var13 , 2*np.pi*series_t) # omega*t [f,t]
    var13 = None
    var15 = np.repeat(var14[np.newaxis,:,:], num_nodes, axis=0) # omega*t [n,f,t]
    var14 = None
    var16 = np.einsum('f,nft->nft', np.sqrt(2*delta_freq), np.cos(var15+var12), optimize=True)
    var12 = None
    var15 = None
    series_v = np.einsum('fmn,nft->mt', np.abs(Gmn_v), var16, optimize=True)
    var16 = None
    
    # Time Series "series_w"
    var12 = np.repeat(PSI_w[:,:,np.newaxis], series_N, axis=2) # PSI_w[n,f,t]
    var13 = np.repeat(freq[:,np.newaxis], series_N, axis=1) # freq[f,t]
    var14 = np.einsum('ft,t->ft', var13 , 2*np.pi*series_t) # omega*t [f,t]
    var13 = None
    var15 = np.repeat(var14[np.newaxis,:,:], num_nodes, axis=0) # omega*t [n,f,t]
    var14 = None
    var16 = np.einsum('f,nft->nft', np.sqrt(2*delta_freq), np.cos(var15+var12), optimize=True)
    var12 = None
    var15 = None
    series_w = np.einsum('fmn,nft->mt', np.abs(Gmn_w), var16, optimize=True)
    var16 = None
    # =============================================================================
    series_u = series_U - np.repeat(V[:, np.newaxis], series_N, axis=1)

    return {'windspeed':np.array([series_U, series_u, series_v, series_w ]),
            'timepoints':series_t,
            'autospec_nondim':autospec_nondim, 
            'cospec':S_aa_reshaped,
            'freq':freq,
            'delta_xyz':np.array([delta_x, delta_y, delta_z])}

# #######################################################################################################################
# # # TESTING
# #######################################################################################################################
# node_coor_wind = np.array([[    0.        ,     0.        ,    14.5       ],
#        [  -78.7191792 ,    61.66812383,    14.5       ],
#        [ -158.65589535,   121.74963582,    14.5       ],
#        [ -239.77817482,   180.22050416,    14.5       ],
#        [ -322.05356978,   237.05734129,    14.5       ],
#        [ -405.44917117,   292.23741323,    14.5       ],
#        [ -489.93162187,   345.73864869,    14.5       ],
#        [ -575.46713001,   397.53964789,    14.5       ],
#        [ -662.02148254,   447.61969111,    14.5       ],
#        [ -749.56005887,   495.95874702,    14.5       ],
#        [ -838.04784473,   542.53748062,    14.5       ],
#        [ -927.4494462 ,   587.33726105,    14.5       ],
#        [-1017.72910382,   630.34016899,    14.5       ],
#        [-1108.85070693,   671.52900386,    14.5       ],
#        [-1200.7778081 ,   710.88729066,    14.5       ],
#        [-1293.47363773,   748.39928661,    14.5       ],
#        [-1386.90111872,   784.04998742,    14.5       ],
#        [-1481.02288132,   817.82513327,    14.5       ],
#        [-1575.80127807,   849.71121455,    14.5       ],
#        [-1671.19839889,   879.69547727,    14.5       ],
#        [-1767.17608619,   907.76592811,    14.5       ],
#        [-1863.69595019,   933.91133927,    14.5       ],
#        [-1960.71938423,   958.12125293,    14.5       ],
#        [-2058.20758021,   980.38598545,    14.5       ],
#        [-2156.12154418,  1000.69663124,    14.5       ],
#        [-2254.42211184,  1019.0450663 ,    14.5       ],
#        [-2353.06996427,  1035.42395151,    14.5       ],
#        [-2452.02564366,  1049.82673553,    14.5       ],
#        [-2551.24956905,  1062.24765744,    14.5       ],
#        [-2650.7020522 ,  1072.68174903,    14.5       ],
#        [-2750.34331343,  1081.12483682,    14.5       ],
#        [-2850.13349757,  1087.57354367,    14.5       ],
#        [-2950.03268988,  1092.02529019,    14.5       ],
#        [-3050.00093201,  1094.47829574,    14.5       ],
#        [-3149.998238  ,  1094.93157916,    14.5       ],
#        [-3249.98461026,  1093.38495912,    14.5       ],
#        [-3349.92005557,  1089.83905427,    14.5       ],
#        [-3449.7646011 ,  1084.29528291,    14.5       ],
#        [-3549.47831034,  1076.75586249,    14.5       ],
#        [-3649.02129915,  1067.22380866,    14.5       ],
#        [-3748.35375165,  1055.70293412,    14.5       ],
#        [-3847.4359362 ,  1042.19784707,    14.5       ],
#        [-3946.22822124,  1026.71394937,    14.5       ],
#        [-4044.69109117,  1009.25743436,    14.5       ],
#        [-4142.78516216,   989.83528442,    14.5       ],
#        [-4240.47119788,   968.45526815,    14.5       ],
#        [-4337.71012523,   945.12593728,    14.5       ],
#        [-4434.46304993,   919.85662321,    14.5       ],
#        [-4530.6912721 ,   892.65743335,    14.5       ],
#        [-4626.35630173,   863.539247  ,    14.5       ],
#        [-4721.41987409,   832.51371106,    14.5       ]])
# V = np.array([40.38529693, 40.38529693, 40.38529693, 40.38529693, 40.38529693,
#        40.38529693, 40.38529693, 40.38529693, 40.38529693, 40.38529693,
#        40.38529693, 40.38529693, 40.38529693, 40.38529693, 40.38529693,
#        40.38529693, 40.38529693, 40.38529693, 40.38529693, 40.38529693,
#        40.38529693, 40.38529693, 40.38529693, 40.38529693, 40.38529693,
#        40.38529693, 40.38529693, 40.38529693, 40.38529693, 40.38529693,
#        40.38529693, 40.38529693, 40.38529693, 40.38529693, 40.38529693,
#        40.38529693, 40.38529693, 40.38529693, 40.38529693, 40.38529693,
#        40.38529693, 40.38529693, 40.38529693, 40.38529693, 40.38529693,
#        40.38529693, 40.38529693, 40.38529693, 40.38529693, 40.38529693,
#        40.38529693])
# Ai = np.array([6.8, 9.4, 9.4])
# Cij = np.array([ 3. , 10. , 10. ,  6. ,  6.5,  6.5,  3. ,  6.5,  3. ])
# I = np.array([[0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077],
#        [0.14 , 0.119, 0.077]])
# iLj = np.array([[111.79191628, 111.79191628, 111.79191628, 111.79191628,
#         111.79191628, 111.79191628, 111.79191628, 111.79191628,
#         111.79191628, 111.79191628, 111.79191628, 111.79191628,
#         111.79191628, 111.79191628, 111.79191628, 111.79191628,
#         111.79191628, 111.79191628, 111.79191628, 111.79191628,
#         111.79191628, 111.79191628, 111.79191628, 111.79191628,
#         111.79191628, 111.79191628, 111.79191628, 111.79191628,
#         111.79191628, 111.79191628, 111.79191628, 111.79191628,
#         111.79191628, 111.79191628, 111.79191628, 111.79191628,
#         111.79191628, 111.79191628, 111.79191628, 111.79191628,
#         111.79191628, 111.79191628, 111.79191628, 111.79191628,
#         111.79191628, 111.79191628, 111.79191628, 111.79191628,
#         111.79191628, 111.79191628, 111.79191628],
#        [ 37.26397209,  37.26397209,  37.26397209,  37.26397209,
#          37.26397209,  37.26397209,  37.26397209,  37.26397209,
#          37.26397209,  37.26397209,  37.26397209,  37.26397209,
#          37.26397209,  37.26397209,  37.26397209,  37.26397209,
#          37.26397209,  37.26397209,  37.26397209,  37.26397209,
#          37.26397209,  37.26397209,  37.26397209,  37.26397209,
#          37.26397209,  37.26397209,  37.26397209,  37.26397209,
#          37.26397209,  37.26397209,  37.26397209,  37.26397209,
#          37.26397209,  37.26397209,  37.26397209,  37.26397209,
#          37.26397209,  37.26397209,  37.26397209,  37.26397209,
#          37.26397209,  37.26397209,  37.26397209,  37.26397209,
#          37.26397209,  37.26397209,  37.26397209,  37.26397209,
#          37.26397209,  37.26397209,  37.26397209],
#        [ 22.35838326,  22.35838326,  22.35838326,  22.35838326,
#          22.35838326,  22.35838326,  22.35838326,  22.35838326,
#          22.35838326,  22.35838326,  22.35838326,  22.35838326,
#          22.35838326,  22.35838326,  22.35838326,  22.35838326,
#          22.35838326,  22.35838326,  22.35838326,  22.35838326,
#          22.35838326,  22.35838326,  22.35838326,  22.35838326,
#          22.35838326,  22.35838326,  22.35838326,  22.35838326,
#          22.35838326,  22.35838326,  22.35838326,  22.35838326,
#          22.35838326,  22.35838326,  22.35838326,  22.35838326,
#          22.35838326,  22.35838326,  22.35838326,  22.35838326,
#          22.35838326,  22.35838326,  22.35838326,  22.35838326,
#          22.35838326,  22.35838326,  22.35838326,  22.35838326,
#          22.35838326,  22.35838326,  22.35838326],
#        [ 27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907],
#        [ 27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907,  27.94797907,
#          27.94797907,  27.94797907,  27.94797907],
#        [  9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302],
#        [  9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302,   9.31599302,
#           9.31599302,   9.31599302,   9.31599302],
#        [  6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202],
#        [  6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202,   6.21066202,
#           6.21066202,   6.21066202,   6.21066202]])
# T = 604.0
# sample_freq = 4.0
# spectrum_type = 2
#
# # Running the function (optional):
# windfield = wind_field_3D_func(node_coor_wind, V, Ai, Cij, I, iLj, T, sample_freq, spectrum_type)
#
#
# # For profiling (testing speed), install the package "line_profiler" (in Anaconda prompt, type: conda install line_profiler). Then the following can be used:
# # Running the profiler:
# %load_ext line_profiler
# %lprun -f wind_field_3D_func wind_field_3D_func(node_coor_wind, V, Ai, Cij, I, iLj, T, sample_freq, spectrum_type)
#
#
