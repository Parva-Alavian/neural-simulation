import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
from pytictoc import TicToc
import math
from random import gauss
from random import seed
from pandas import Series
import sdeint
from numpy import linalg as LA

def g_fun(z,t):
    sigma_a = 2.5
    sigma_b = 2.5
    sigma_c = 2.5
    sigma_d = 2.5
    sigma_e = 2.5
    sigma_f = 2.5
    sigma_g = 2.5
    
    #
    #sigma_a = 0.00
    #sigma_b = 0.00
    #sigma_c = 0.00
    #sigma_d = 0.00
    #sigma_e = 0.00
    #sigma_f = 0.00
    #sigma_g = 0.00
    
    tau_y = 0.01
    return np.diag([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,sigma_a*tau_y,sigma_b*tau_y,sigma_c*tau_y,sigma_d*tau_y,sigma_e*tau_y,sigma_f*tau_y,sigma_g*tau_y])


def model_MW1(z,t):

    
    ## ...................................................
    
    tau_a = 0.03
    tau_b = 0.03
    tau_c = 0.0025
    tau_Aa = 0.001
    tau_Ab = 0.001
    tau_r = 0.001
    tau_n = 0.001
    tau_sa = 0.0075
    tau_sb = 0.0075
    tau_va = 0.0075
    tau_vb = 0.0075
    
    gamma_a = 2.564
    gamma_b = 2.564
    gamma_Aa = 2.0
    gamma_Ab = 2.0
    gamma_c = 4.0
    
    gamma_sa = 3.0
    gamma_sb = 3.0
    gamma_va = 3.0
    gamma_vb = 3.0
    
    gamma_ra = 1.0
    gamma_rb = 1.0
    gamma_rc = 1.0
    gamma_rs = 1.0
    gamma_v = 1.0   
    ## ...................................................
    
        
    g_I = 4
    c_1 = 615
    c_0 = 177
    r_0 = 5.5
    
    a = 135
    b = 54
    d = 0.308
    
    J_s = 0.3213
    J_c = 0.0107
    J_ei = -0.36
    J_ii = -0.12
    
    J_0 = 0.2112
   
    J_s = 0.4813
    #J_s = 0.6813
    
    eta=tau_c*gamma_c*c_1/(g_I-J_ii*tau_c*gamma_c*c_1)
    J_ie = (J_0-J_s-J_c)/(2*J_ei*eta)
    
    J_se = 0.45   
    J_ce = 0.0
    J_sp = 0.1
    
    J_es = -0.55
    
    #J_pv = 0.25
    #J_sv = 0.33
    
    J_pv = 0.45
    J_sv = -0.15
    
    J_vs = -0.65
            
    
    J_Aa = 3.8
    J_Ab = 1.75
    
    J_Ia = 3.2
    J_Ib = 3.2
    
        
    Iback_a = 0.375
    Iback_b = 0.375
    Iback_c = 0.285    
    Iback_d = 0.12
    Iback_e = 0.12
    Iback_f = 0.0
    Iback_g = 0.0
        
    
    idx=(t>4 and t<8)
    idy=(t>12 and t<16)
    
    I_va = 0.1*idy
    I_vb = 0.1*idy
    
    I_a = 0.08*idx
    I_b = 0.0*idx
    I_c = 0.0*idx
    
        
    p1 = 1.0
    p2 = 1.0
         
    
    #-------------------------------------------------------------
    dz=np.zeros(23)
    
    sa    = z[0]
    sb    = z[1]
    sc    = z[2]
    s_Aa  = z[3]
    s_Ab  = z[4]
    st_a  = z[5]
    st_b  = z[6]
    s_va  = z[7]
    s_vb  = z[8]
    r_a   = z[9]
    r_b   = z[10]
    r_c   = z[11]
    r_d   = z[12]
    r_e   = z[13]
    r_f   = z[14]
    r_g   = z[15]
    y_a   = z[16]
    y_b   = z[17]
    y_c   = z[18]
    y_d   = z[19]
    y_e   = z[20]
    y_f   = z[21]
    y_g   = z[22]
    
       
    
    input_a = J_s*sa+J_c*sb+J_ei*sc+J_Aa*s_Aa+J_Ab*s_Ab+J_es*st_a+I_a+Iback_a+y_a
    input_b = J_c*sa+J_s*sb+J_ei*sc+J_Aa*s_Ab+J_Ab*s_Aa+J_es*st_b+I_b+Iback_b+y_b
    input_c = J_ie*sa+J_ie*sb+J_ii*sc+J_Ia*s_Aa+J_Ib*s_Ab+I_c+Iback_c+y_c
    
    input_sa = J_se*sa+J_ce*sb+J_sp*s_Aa+J_vs*s_va+Iback_d+y_d
    input_sb = J_ce*sa+J_se*sb+J_sp*s_Ab+J_vs*s_vb+Iback_e+y_e
    
    input_va = J_pv*sa+J_sv*st_a+Iback_f+I_va+y_f
    input_vb = J_pv*sb+J_sv*st_b+Iback_g+I_vb+y_g
    
    g1=1
    
    phi_a = g1*p1*(a*input_a-b)/(1-np.exp(-d*(a*input_a-b)))    
    phi_b = g1*p2*(a*input_b-b)/(1-np.exp(-d*(a*input_b-b)))

    g1=int(input_c>=(c_0-r_0*g_I)/c_1)
    phi_c = g1*(((c_1*input_c-c_0)/g_I)+r_0)
    
    g1=int(input_sa>=(c_0-r_0*g_I)/c_1)
    phi_sa = g1*(((c_1*input_sa-c_0)/g_I)+r_0)
    
    g1=int(input_sb>=(c_0-r_0*g_I)/c_1)
    phi_sb = g1*(((c_1*input_sb-c_0)/g_I)+r_0) 
    
    g1=int(input_va>=(c_0-r_0*g_I)/c_1)
    phi_va = g1*(((c_1*input_va-c_0)/g_I)+r_0)
    
    g1=int(input_vb>=(c_0-r_0*g_I)/c_1)
    phi_vb = g1*(((c_1*input_vb-c_0)/g_I)+r_0) 


    dsa = (-(sa/tau_a)+(1-sa)*gamma_a*r_a)
    dsb = (-(sb/tau_b)+(1-sb)*gamma_b*r_b)
    
    ds_Aa = (-(s_Aa/tau_Aa)+gamma_Aa*r_a)
    ds_Ab = (-(s_Ab/tau_Ab)+gamma_Ab*r_b)
    
    dst_a = (-(st_a/tau_sa)+gamma_sa*r_d)
    dst_b = (-(st_b/tau_sb)+gamma_sb*r_e)
    
    ds_va = (-(s_va/tau_va)+gamma_va*r_f)
    ds_vb = (-(s_vb/tau_vb)+gamma_vb*r_g)
    
    dsc = (-(sc/tau_c)+gamma_c*r_c)

    dr_a = ((phi_a-r_a)/tau_r)*gamma_ra
    dr_b = ((phi_b-r_b)/tau_r)*gamma_rb
    dr_c = ((phi_c-r_c)/tau_r)*gamma_rc
    dr_d = ((phi_sa-r_d)/tau_r)*gamma_rs
    dr_e = ((phi_sb-r_e)/tau_r)*gamma_rs
    dr_f = ((phi_va-r_f)/tau_r)*gamma_v
    dr_g = ((phi_vb-r_g)/tau_r)*gamma_v

    dy_a = -(y_a/tau_n)
    dy_b = -(y_b/tau_n)
    dy_c = -(y_c/tau_n)
    dy_d = -(y_d/tau_n)
    dy_e = -(y_e/tau_n)
    dy_f = -(y_f/tau_n)
    dy_g = -(y_g/tau_n)
    
    
        
    dz[0]  = dsa
    dz[1]  = dsb
    dz[2]  = dsc
    dz[3]  = ds_Aa
    dz[4]  = ds_Ab
    dz[5]  = dst_a
    dz[6]  = dst_b
    dz[7]  = ds_va
    dz[8]  = ds_vb
    dz[9]  = dr_a
    dz[10]  = dr_b
    dz[11]  = dr_c
    dz[12] = dr_d
    dz[13] = dr_e
    dz[14] = dr_f
    dz[15] = dr_g
    dz[16] = dy_a
    dz[17] = dy_b
    dz[18] = dy_c
    dz[19] = dy_d
    dz[20] = dy_e
    dz[21] = dy_f
    dz[22] = dy_g
        
    
    
    return dz
    
    
    
    
def model_MW2(z,t):

    ## ...................................................
    
    tau_a = 0.03
    tau_b = 0.03
    tau_c = 0.0025
    tau_Aa = 0.001
    tau_Ab = 0.001
    tau_r = 0.001
    tau_n = 0.001
    tau_sa = 0.0075
    tau_sb = 0.0075
    tau_va = 0.0075
    tau_vb = 0.0075
    
    gamma_a = 2.564
    gamma_b = 2.564
    gamma_Aa = 2.0
    gamma_Ab = 2.0
    gamma_c = 4.0
    
    gamma_sa = 3.0
    gamma_sb = 3.0
    gamma_va = 3.0
    gamma_vb = 3.0
    
    gamma_ra = 1.0
    gamma_rb = 1.0
    gamma_rc = 1.0
    gamma_rs = 1.0
    gamma_v = 1.0   
    ## ...................................................
    
        
    g_I = 4
    c_1 = 615
    c_0 = 177
    r_0 = 5.5
    
    a = 135
    b = 54
    d = 0.308
    
    J_s = 0.3213
    J_c = 0.0107
    J_ei = -0.36
    J_ii = -0.12
    
    J_0 = 0.2112
   
    J_s = 0.4813
    #J_s = 0.6813
    
    eta=tau_c*gamma_c*c_1/(g_I-J_ii*tau_c*gamma_c*c_1)
    J_ie = (J_0-J_s-J_c)/(2*J_ei*eta)
    
    J_se = 0.45   
    J_ce = 0.0
    J_sp = 0.1
    
    J_es = -0.55
    
    #J_pv = 0.25
    #J_sv = 0.33
    
    J_pv = 0.45
    J_sv = -0.15
    
    J_vs = -0.65
            
    
    J_Aa = 3.8
    J_Ab = 1.75
    
    J_Ia = 3.2
    J_Ib = 3.2
    
        
    Iback_a = 0.375
    Iback_b = 0.375
    Iback_c = 0.285    
    Iback_d = 0.12
    Iback_e = 0.12
    Iback_f = 0.0
    Iback_g = 0.0    
    
           
    idx=(t>4 and t<8)
    idy=(t>12 and t<16)
    
    I_va = 0.1*idy
    I_vb = 0.1*idy
    
    I_a = 0.06*idx
    I_b = 0.0*idx
    I_c = 0.0*idx
    
        
    p1 = 1.0
    p2 = 1.0
              
    
    #-------------------------------------------------------------
    dz=np.zeros(23)
    
    sa    = z[0]
    sb    = z[1]
    sc    = z[2]
    s_Aa  = z[3]
    s_Ab  = z[4]
    st_a  = z[5]
    st_b  = z[6]
    s_va  = z[7]
    s_vb  = z[8]
    r_a   = z[9]
    r_b   = z[10]
    r_c   = z[11]
    r_d   = z[12]
    r_e   = z[13]
    r_f   = z[14]
    r_g   = z[15]
    y_a   = z[16]
    y_b   = z[17]
    y_c   = z[18]
    y_d   = z[19]
    y_e   = z[20]
    y_f   = z[21]
    y_g   = z[22]
    
       
    
    input_a = J_s*sa+J_c*sb+J_ei*sc+J_Aa*s_Aa+J_Ab*s_Ab+J_es*st_a+I_a+Iback_a+y_a
    input_b = J_c*sa+J_s*sb+J_ei*sc+J_Aa*s_Ab+J_Ab*s_Aa+J_es*st_b+I_b+Iback_b+y_b
    input_c = J_ie*sa+J_ie*sb+J_ii*sc+J_Ia*s_Aa+J_Ib*s_Ab+I_c+Iback_c+y_c
    
    input_sa = J_se*sa+J_ce*sb+J_sp*s_Aa+J_vs*s_va+Iback_d+y_d
    input_sb = J_ce*sa+J_se*sb+J_sp*s_Ab+J_vs*s_vb+Iback_e+y_e
    
    input_va = J_pv*sa+J_sv*st_a+Iback_f+I_va+y_f
    input_vb = J_pv*sb+J_sv*st_b+Iback_g+I_vb+y_g
    
    g1=1
    
    phi_a = g1*p1*(a*input_a-b)/(1-np.exp(-d*(a*input_a-b)))    
    phi_b = g1*p2*(a*input_b-b)/(1-np.exp(-d*(a*input_b-b)))

    g1=int(input_c>=(c_0-r_0*g_I)/c_1)
    phi_c = g1*(((c_1*input_c-c_0)/g_I)+r_0)
    
    g1=int(input_sa>=(c_0-r_0*g_I)/c_1)
    phi_sa = g1*(((c_1*input_sa-c_0)/g_I)+r_0)
    
    g1=int(input_sb>=(c_0-r_0*g_I)/c_1)
    phi_sb = g1*(((c_1*input_sb-c_0)/g_I)+r_0) 
    
    g1=int(input_va>=(c_0-r_0*g_I)/c_1)
    phi_va = g1*(((c_1*input_va-c_0)/g_I)+r_0)
    
    g1=int(input_vb>=(c_0-r_0*g_I)/c_1)
    phi_vb = g1*(((c_1*input_vb-c_0)/g_I)+r_0) 


    dsa = (-(sa/tau_a)+(1-sa)*gamma_a*r_a)
    dsb = (-(sb/tau_b)+(1-sb)*gamma_b*r_b)
    
    ds_Aa = (-(s_Aa/tau_Aa)+gamma_Aa*r_a)
    ds_Ab = (-(s_Ab/tau_Ab)+gamma_Ab*r_b)
    
    dst_a = (-(st_a/tau_sa)+gamma_sa*r_d)
    dst_b = (-(st_b/tau_sb)+gamma_sb*r_e)
    
    ds_va = (-(s_va/tau_va)+gamma_va*r_f)
    ds_vb = (-(s_vb/tau_vb)+gamma_vb*r_g)
    
    dsc = (-(sc/tau_c)+gamma_c*r_c)

    dr_a = ((phi_a-r_a)/tau_r)*gamma_ra
    dr_b = ((phi_b-r_b)/tau_r)*gamma_rb
    dr_c = ((phi_c-r_c)/tau_r)*gamma_rc
    dr_d = ((phi_sa-r_d)/tau_r)*gamma_rs
    dr_e = ((phi_sb-r_e)/tau_r)*gamma_rs
    dr_f = ((phi_va-r_f)/tau_r)*gamma_v
    dr_g = ((phi_vb-r_g)/tau_r)*gamma_v

    dy_a = -(y_a/tau_n)
    dy_b = -(y_b/tau_n)
    dy_c = -(y_c/tau_n)
    dy_d = -(y_d/tau_n)
    dy_e = -(y_e/tau_n)
    dy_f = -(y_f/tau_n)
    dy_g = -(y_g/tau_n)
    
    
        
    dz[0]  = dsa
    dz[1]  = dsb
    dz[2]  = dsc
    dz[3]  = ds_Aa
    dz[4]  = ds_Ab
    dz[5]  = dst_a
    dz[6]  = dst_b
    dz[7]  = ds_va
    dz[8]  = ds_vb
    dz[9]  = dr_a
    dz[10]  = dr_b
    dz[11]  = dr_c
    dz[12] = dr_d
    dz[13] = dr_e
    dz[14] = dr_f
    dz[15] = dr_g
    dz[16] = dy_a
    dz[17] = dy_b
    dz[18] = dy_c
    dz[19] = dy_d
    dz[20] = dy_e
    dz[21] = dy_f
    dz[22] = dy_g
        
    
    
    return dz
    
    
    
    
def model_MW3(z,t):

    ## ...................................................
    
    tau_a = 0.03
    tau_b = 0.03
    tau_c = 0.0025
    tau_Aa = 0.001
    tau_Ab = 0.001
    tau_r = 0.001
    tau_n = 0.001
    tau_sa = 0.0075
    tau_sb = 0.0075
    tau_va = 0.0075
    tau_vb = 0.0075
    
    gamma_a = 2.564
    gamma_b = 2.564
    gamma_Aa = 2.0
    gamma_Ab = 2.0
    gamma_c = 4.0
    
    gamma_sa = 3.0
    gamma_sb = 3.0
    gamma_va = 3.0
    gamma_vb = 3.0
    
    gamma_ra = 1.0
    gamma_rb = 1.0
    gamma_rc = 1.0
    gamma_rs = 1.0
    gamma_v = 1.0   
    ## ...................................................
    
        
    g_I = 4
    c_1 = 615
    c_0 = 177
    r_0 = 5.5
    
    a = 135
    b = 54
    d = 0.308
    
    J_s = 0.3213
    J_c = 0.0107
    J_ei = -0.36
    J_ie = 0.15
    J_ii = -0.12
    
    J_0 = 0.2112
   
    J_s = 0.4813
    #J_s = 0.6813
    
    J_se = 0.45   
    J_ce = 0.0
    J_sp = 0.1
    
    J_es = -0.55
    
    #J_pv = 0.25
    #J_sv = 0.33
    
    J_pv = 0.45
    J_sv = -0.15
    
    J_vs = -0.65
            
    
    J_Aa = 3.8
    J_Ab = 1.75
    
    J_Ia = 3.2
    J_Ib = 3.2
    

    Iback_a = 0.375
    Iback_b = 0.375
    Iback_c = 0.285    
    Iback_d = 0.12
    Iback_e = 0.12
    Iback_f = 0.0
    Iback_g = 0.0
    
           
    idx=(t>4 and t<8)
    idy=(t>12 and t<16)
    
    I_va = 0.1*idy
    I_vb = 0.1*idy
    
    I_a = 0.0*idx
    I_b = 0.0*idx
    I_c = 0.0*idx
    
        
    p1 = 1.0
    p2 = 1.0
    
           
    eta=tau_c*gamma_c*c_1/(g_I-J_ii*tau_c*gamma_c*c_1)
    J_ie = (J_0-J_s-J_c)/(2*J_ei*eta)
    #-------------------------------------------------------------
    dz=np.zeros(23)
    
    sa    = z[0]
    sb    = z[1]
    sc    = z[2]
    s_Aa  = z[3]
    s_Ab  = z[4]
    st_a  = z[5]
    st_b  = z[6]
    s_va  = z[7]
    s_vb  = z[8]
    r_a   = z[9]
    r_b   = z[10]
    r_c   = z[11]
    r_d   = z[12]
    r_e   = z[13]
    r_f   = z[14]
    r_g   = z[15]
    y_a   = z[16]
    y_b   = z[17]
    y_c   = z[18]
    y_d   = z[19]
    y_e   = z[20]
    y_f   = z[21]
    y_g   = z[22]
    
       
    
    input_a = J_s*sa+J_c*sb+J_ei*sc+J_Aa*s_Aa+J_Ab*s_Ab+J_es*st_a+I_a+Iback_a+y_a
    input_b = J_c*sa+J_s*sb+J_ei*sc+J_Aa*s_Ab+J_Ab*s_Aa+J_es*st_b+I_b+Iback_b+y_b
    input_c = J_ie*sa+J_ie*sb+J_ii*sc+J_Ia*s_Aa+J_Ib*s_Ab+I_c+Iback_c+y_c
    
    input_sa = J_se*sa+J_ce*sb+J_sp*s_Aa+J_vs*s_va+Iback_d+y_d
    input_sb = J_ce*sa+J_se*sb+J_sp*s_Ab+J_vs*s_vb+Iback_e+y_e
    
    input_va = J_pv*sa+J_sv*st_a+Iback_f+I_va+y_f
    input_vb = J_pv*sb+J_sv*st_b+Iback_g+I_vb+y_g
    
    g1=1
    
    phi_a = g1*p1*(a*input_a-b)/(1-np.exp(-d*(a*input_a-b)))    
    phi_b = g1*p2*(a*input_b-b)/(1-np.exp(-d*(a*input_b-b)))

    g1=int(input_c>=(c_0-r_0*g_I)/c_1)
    phi_c = g1*(((c_1*input_c-c_0)/g_I)+r_0)
    
    g1=int(input_sa>=(c_0-r_0*g_I)/c_1)
    phi_sa = g1*(((c_1*input_sa-c_0)/g_I)+r_0)
    
    g1=int(input_sb>=(c_0-r_0*g_I)/c_1)
    phi_sb = g1*(((c_1*input_sb-c_0)/g_I)+r_0) 
    
    g1=int(input_va>=(c_0-r_0*g_I)/c_1)
    phi_va = g1*(((c_1*input_va-c_0)/g_I)+r_0)
    
    g1=int(input_vb>=(c_0-r_0*g_I)/c_1)
    phi_vb = g1*(((c_1*input_vb-c_0)/g_I)+r_0) 


    dsa = (-(sa/tau_a)+(1-sa)*gamma_a*r_a)
    dsb = (-(sb/tau_b)+(1-sb)*gamma_b*r_b)
    
    ds_Aa = (-(s_Aa/tau_Aa)+gamma_Aa*r_a)
    ds_Ab = (-(s_Ab/tau_Ab)+gamma_Ab*r_b)
    
    dst_a = (-(st_a/tau_sa)+gamma_sa*r_d)
    dst_b = (-(st_b/tau_sb)+gamma_sb*r_e)
    
    ds_va = (-(s_va/tau_va)+gamma_va*r_f)
    ds_vb = (-(s_vb/tau_vb)+gamma_vb*r_g)
    
    dsc = (-(sc/tau_c)+gamma_c*r_c)

    dr_a = ((phi_a-r_a)/tau_r)*gamma_ra
    dr_b = ((phi_b-r_b)/tau_r)*gamma_rb
    dr_c = ((phi_c-r_c)/tau_r)*gamma_rc
    dr_d = ((phi_sa-r_d)/tau_r)*gamma_rs
    dr_e = ((phi_sb-r_e)/tau_r)*gamma_rs
    dr_f = ((phi_va-r_f)/tau_r)*gamma_v
    dr_g = ((phi_vb-r_g)/tau_r)*gamma_v

    dy_a = -(y_a/tau_n)
    dy_b = -(y_b/tau_n)
    dy_c = -(y_c/tau_n)
    dy_d = -(y_d/tau_n)
    dy_e = -(y_e/tau_n)
    dy_f = -(y_f/tau_n)
    dy_g = -(y_g/tau_n)
    
    
        
    dz[0]  = dsa
    dz[1]  = dsb
    dz[2]  = dsc
    dz[3]  = ds_Aa
    dz[4]  = ds_Ab
    dz[5]  = dst_a
    dz[6]  = dst_b
    dz[7]  = ds_va
    dz[8]  = ds_vb
    dz[9]  = dr_a
    dz[10]  = dr_b
    dz[11]  = dr_c
    dz[12] = dr_d
    dz[13] = dr_e
    dz[14] = dr_f
    dz[15] = dr_g
    dz[16] = dy_a
    dz[17] = dy_b
    dz[18] = dy_c
    dz[19] = dy_d
    dz[20] = dy_e
    dz[21] = dy_f
    dz[22] = dy_g
        
    
    
    return dz        


def Ejac_MW1(I,x,y,z):

    #-------------------------------------------------------------
   
    phi_a = I[0]
    phi_b = I[1]
    phi_c = I[2]
    
    
    tau_a = 0.06
    tau_b = 0.06
    tau_c = 0.005
    tau_Aa = 0.002
    tau_Ab = 0.002
    tau_r = 0.002
    tau_n = 0.002
    
    gamma_a = 1.282
    gamma_b = 1.282
    gamma_Aa = 1.0
    gamma_Ab = 1.0
    gamma_c = 2.0
    
    #gamma_c = 0.1
    
    g_I = 4
    c_1 = 615
    c_0 = 177
    r_0 = 5.5
    
    a = 135
    b = 54
    d = 0.308
    
    J_s = 0.3213
    J_c = 0.0107
    J_ei = -0.31
    J_ie = 0.15
    J_ii = -0.12
    
    J_ei = -0.41
    J_ei = -0.40
        
    J_0 = 0.2112
    
    J_s = 0.6813
    
    J_Aa = x
    J_Ab = z
    
    J_Ia = y
    J_Ib = y
    
        
    p1 = 1.0
    p2 = 1.0 
    
    Iback_a = 0.3294
    Iback_b = 0.3294
    Iback_c = 0.26 
    
    eta=tau_c*gamma_c*c_1/(g_I-J_ii*tau_c*gamma_c*c_1)
    J_ie = (J_0-J_s-J_c)/(2*J_ei*eta)
    #-------------------------------------------------------------
    dz=np.zeros((5,5))
    
    sa    = gamma_a*phi_a/(gamma_a*phi_a+1/tau_a)
    sb    = gamma_b*phi_b/(gamma_a*phi_b+1/tau_b)
    sc    = gamma_c*phi_c*tau_c
    s_Aa    = gamma_Aa*phi_a*tau_Aa
    s_Ab    = gamma_Ab*phi_b*tau_Aa
       
    
    input_a = J_s*sa+J_c*sb+J_ei*sc+J_Aa*s_Aa+J_Ab*s_Ab+Iback_a
    input_b = J_c*sa+J_s*sb+J_ei*sc+J_Ab*s_Aa+J_Aa*s_Ab+Iback_b
    input_c = J_ie*sa+J_ie*sb+J_ii*sc+J_Ia*s_Aa+J_Ib*s_Ab+Iback_c
    
    G1=(1-np.exp(-d*(a*input_a-b)))
    G2=(1-np.exp(-d*(a*input_b-b)))
    
    g1=int(input_c>=(c_0-r_0*g_I)/c_1)
        
    dz[0,0] = -(gamma_a*phi_a+1/tau_a)+(1-sa)*gamma_a*(a*G1*J_s-(a*input_a-b)*d*a*J_s*np.exp(-d*(a*input_a-b)))/(G1*G1)
    dz[0,1] = (1-sa)*gamma_a*(a*G1*J_c-(a*input_a-b)*d*a*J_c*np.exp(-d*(a*input_a-b)))/(G1*G1)
    dz[0,2] = (1-sa)*gamma_a*(a*G1*J_Aa-(a*input_a-b)*d*a*J_Aa*np.exp(-d*(a*input_a-b)))/(G1*G1)
    dz[0,3] = (1-sa)*gamma_a*(a*G1*J_Ab-(a*input_a-b)*d*a*J_Ab*np.exp(-d*(a*input_a-b)))/(G1*G1)
    dz[0,4] = (1-sa)*gamma_a*(a*G1*J_ei-(a*input_a-b)*d*a*J_ei*np.exp(-d*(a*input_a-b)))/(G1*G1)
    
    dz[1,0] = (1-sb)*gamma_b*(a*G2*J_c-(a*input_b-b)*d*a*J_c*np.exp(-d*(a*input_b-b)))/(G2*G2)
    dz[1,1] = -(gamma_b*phi_b+1/tau_b)+(1-sb)*gamma_b*(a*G2*J_s-(a*input_b-b)*d*a*J_s*np.exp(-d*(a*input_b-b)))/(G2*G2)
    dz[1,2] = (1-sb)*gamma_b*(a*G2*J_Ab-(a*input_b-b)*d*a*J_Ab*np.exp(-d*(a*input_b-b)))/(G2*G2)
    dz[1,3] = (1-sb)*gamma_b*(a*G2*J_Aa-(a*input_b-b)*d*a*J_Aa*np.exp(-d*(a*input_b-b)))/(G2*G2)
    dz[1,4] = (1-sb)*gamma_b*(a*G2*J_ei-(a*input_b-b)*d*a*J_ei*np.exp(-d*(a*input_b-b)))/(G2*G2)
    
    dz[2,0] = gamma_Aa*(a*G1*J_s-(a*input_a-b)*d*a*J_s*np.exp(-d*(a*input_a-b)))/(G1*G1)
    dz[2,1] = gamma_Aa*(a*G1*J_c-(a*input_a-b)*d*a*J_c*np.exp(-d*(a*input_a-b)))/(G1*G1)
    dz[2,2] = -1/tau_Aa+gamma_Aa*(a*G1*J_Aa-(a*input_a-b)*d*a*J_Aa*np.exp(-d*(a*input_a-b)))/(G1*G1)
    dz[2,3] = gamma_Aa*(a*G1*J_Ab-(a*input_a-b)*d*a*J_Ab*np.exp(-d*(a*input_a-b)))/(G1*G1)
    dz[2,4] = gamma_Aa*(a*G1*J_ei-(a*input_a-b)*d*a*J_ei*np.exp(-d*(a*input_a-b)))/(G1*G1)
    
    dz[3,0] = gamma_Ab*(a*G2*J_c-(a*input_b-b)*d*a*J_c*np.exp(-d*(a*input_b-b)))/(G2*G2)
    dz[3,1] = gamma_Ab*(a*G2*J_s-(a*input_b-b)*d*a*J_s*np.exp(-d*(a*input_b-b)))/(G2*G2)
    dz[3,2] = gamma_Ab*(a*G2*J_Ab-(a*input_b-b)*d*a*J_Ab*np.exp(-d*(a*input_b-b)))/(G2*G2)
    dz[3,3] = -1/tau_Ab+gamma_Ab*(a*G2*J_Aa-(a*input_b-b)*d*a*J_Aa*np.exp(-d*(a*input_b-b)))/(G2*G2)
    dz[3,4] = gamma_Ab*(a*G2*J_ei-(a*input_b-b)*d*a*J_ei*np.exp(-d*(a*input_b-b)))/(G2*G2)
    
    dz[4,0] = gamma_c*(g1*c_1/g_I)*J_ie
    dz[4,1] = gamma_c*(g1*c_1/g_I)*J_ie
    dz[4,2] = gamma_c*(g1*c_1/g_I)*J_Ia
    dz[4,3] = gamma_c*(g1*c_1/g_I)*J_Ib
    dz[4,4] = -1/tau_c+gamma_c*(g1*c_1/g_I)*J_ii
    
    
    w,v=LA.eig(dz)
    
    return w

teps = TicToc()
teps.tic()


z1=[0.05,0.05,0.03,0.05,0.05,0.05,0.05,0.05,0.05,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

Np=20
nn1=200001

#Np=100
#nn1=1001

Te=Np


t = np.linspace(0,Np,nn1)
T = np.linspace(0,Np*1000,nn1)

clr=['b','m','r']

nn1-1
fig1, axs = plt.subplots(3,2) 
fig2, ax = plt.subplots(3,2)
fig3, ax1 = plt.subplots(3,2)
fig4 = plt.figure(figsize=plt.figaspect(0.5))
ax4 = fig4.add_subplot(111, projection='3d')

for i in range(3):
  if i==0:
     x=sdeint.itoint(model_MW3, g_fun, z1, t)
     
     y_a = x[:,9]
     y_b = x[:,10]
     y_c = x[:,11]
     
     z_a = x[120001:160000,9]
     z_b = x[120001:160000,10]
     z_c = x[120001:160000,11]
     
     w_a = x[160001:nn1-1,9]
     w_b = x[160001:nn1-1,10]
     w_c = x[160001:nn1-1,11]
     
     dt = Np/(nn1-1)
     N = y_a.shape[0] 
     Tn = N * dt
     
     yf_a = np.fft.fft(y_a - y_a.mean())                              # Compute Fourier transform of x
     S_a = 2 * dt ** 2 / Tn * (yf_a * yf_a.conj())                    # Compute spectrum
     S_a = S_a[:int(len(y_a) / 2)]
     
     yf_b = np.fft.fft(y_b - y_b.mean())                              # Compute Fourier transform of x
     S_b = 2 * dt ** 2 / Tn * (yf_b * yf_b.conj())                    # Compute spectrum
     S_b = S_b[:int(len(y_b) / 2)]
     
     yf_c = np.fft.fft(y_c - y_c.mean())                              # Compute Fourier transform of x
     S_c = 2 * dt ** 2 / Tn * (yf_c * yf_c.conj())                    # Compute spectrum
     S_c = S_c[:int(len(y_c) / 2)]
     
     df = 1 / Tn                                                      # Determine frequency resolution
     fNQ = 1 / dt / 2                                                 # Determine Nyquist frequency
     faxis = np.arange(0,fNQ-df,df)                                   # Construct frequency axis
     
     N = z_a.shape[0] 
     Tn = N * dt
     
     zf_a = np.fft.fft(z_a - z_a.mean())                              # Compute Fourier transform of x
     z_a = 2 * dt ** 2 / Tn * (zf_a * zf_a.conj())                    # Compute spectrum
     z_a = z_a[:int(len(z_a) / 2)]
     
     zf_b = np.fft.fft(z_b - z_b.mean())                              # Compute Fourier transform of x
     z_b = 2 * dt ** 2 / Tn * (zf_b * zf_b.conj())                    # Compute spectrum
     z_b = z_b[:int(len(z_b) / 2)]
     
     zf_c = np.fft.fft(z_c - z_c.mean())                              # Compute Fourier transform of x
     z_c = 2 * dt ** 2 / Tn * (zf_c * zf_c.conj())                    # Compute spectrum
     z_c = z_c[:int(len(z_c) / 2)]
     
     df = 1 / Tn                                                      # Determine frequency resolution
     fNQ = 1 / dt / 2                                                 # Determine Nyquist frequency
     faxisz = np.arange(0,fNQ-df,df)                                  # Construct frequency axis
     
     N = w_a.shape[0] 
     Tn = N * dt
     
     wf_a = np.fft.fft(w_a - w_a.mean())                              # Compute Fourier transform of x
     w_a = 2 * dt ** 2 / Tn * (wf_a * wf_a.conj())                    # Compute spectrum
     w_a = w_a[:int(len(w_a) / 2)]
     
     wf_b = np.fft.fft(w_b - w_b.mean())                              # Compute Fourier transform of x
     w_b = 2 * dt ** 2 / Tn * (wf_b * wf_b.conj())                    # Compute spectrum
     w_b = w_b[:int(len(w_b) / 2)]
     
     wf_c = np.fft.fft(w_c - w_c.mean())                              # Compute Fourier transform of x
     w_c = 2 * dt ** 2 / Tn * (wf_c * wf_c.conj())                    # Compute spectrum
     w_c = w_c[:int(len(w_c) / 2)]
     
     df = 1 / Tn                                                      # Determine frequency resolution
     fNQ = 1 / dt / 2                                                 # Determine Nyquist frequency
     faxisw = np.arange(0,fNQ-df,df)                                  # Construct frequency axis
     
     
  elif i==1:
     x=sdeint.itoint(model_MW2, g_fun, z1, t)
     
     y_a = x[:,9]
     y_b = x[:,10]
     y_c = x[:,11]
     
     z_a = x[120001:160000,9]
     z_b = x[120001:160000,10]
     z_c = x[120001:160000,11]
     
     w_a = x[160001:nn1-1,9]
     w_b = x[160001:nn1-1,10]
     w_c = x[160001:nn1-1,11]
     
     dt = Np/(nn1-1)
     N = y_a.shape[0] 
     Tn = N * dt
     
     yf_a = np.fft.fft(y_a - y_a.mean())                              # Compute Fourier transform of x
     S_a = 2 * dt ** 2 / Tn * (yf_a * yf_a.conj())                    # Compute spectrum
     S_a = S_a[:int(len(y_a) / 2)]
     
     yf_b = np.fft.fft(y_b - y_b.mean())                              # Compute Fourier transform of x
     S_b = 2 * dt ** 2 / Tn * (yf_b * yf_b.conj())                    # Compute spectrum
     S_b = S_b[:int(len(y_b) / 2)]
     
     yf_c = np.fft.fft(y_c - y_c.mean())                              # Compute Fourier transform of x
     S_c = 2 * dt ** 2 / Tn * (yf_c * yf_c.conj())                    # Compute spectrum
     S_c = S_c[:int(len(y_c) / 2)]
     
     df = 1 / Tn                                                      # Determine frequency resolution
     fNQ = 1 / dt / 2                                                 # Determine Nyquist frequency
     faxis = np.arange(0,fNQ-df,df)                                   # Construct frequency axis
     
     N = z_a.shape[0] 
     Tn = N * dt
     
     zf_a = np.fft.fft(z_a - z_a.mean())                              # Compute Fourier transform of x
     z_a = 2 * dt ** 2 / Tn * (zf_a * zf_a.conj())                    # Compute spectrum
     z_a = z_a[:int(len(z_a) / 2)]
     
     zf_b = np.fft.fft(z_b - z_b.mean())                              # Compute Fourier transform of x
     z_b = 2 * dt ** 2 / Tn * (zf_b * zf_b.conj())                    # Compute spectrum
     z_b = z_b[:int(len(z_b) / 2)]
     
     zf_c = np.fft.fft(z_c - z_c.mean())                              # Compute Fourier transform of x
     z_c = 2 * dt ** 2 / Tn * (zf_c * zf_c.conj())                    # Compute spectrum
     z_c = z_c[:int(len(z_c) / 2)]
     
     df = 1 / Tn                                                      # Determine frequency resolution
     fNQ = 1 / dt / 2                                                 # Determine Nyquist frequency
     faxisz = np.arange(0,fNQ-df,df)                                  # Construct frequency axis
     
     N = w_a.shape[0] 
     Tn = N * dt
     
     wf_a = np.fft.fft(w_a - w_a.mean())                              # Compute Fourier transform of x
     w_a = 2 * dt ** 2 / Tn * (wf_a * wf_a.conj())                    # Compute spectrum
     w_a = w_a[:int(len(w_a) / 2)]
     
     wf_b = np.fft.fft(w_b - w_b.mean())                              # Compute Fourier transform of x
     w_b = 2 * dt ** 2 / Tn * (wf_b * wf_b.conj())                    # Compute spectrum
     w_b = w_b[:int(len(w_b) / 2)]
     
     wf_c = np.fft.fft(w_c - w_c.mean())                              # Compute Fourier transform of x
     w_c = 2 * dt ** 2 / Tn * (wf_c * wf_c.conj())                    # Compute spectrum
     w_c = w_c[:int(len(w_c) / 2)]
     
     df = 1 / Tn                                                      # Determine frequency resolution
     fNQ = 1 / dt / 2                                                 # Determine Nyquist frequency
     faxisw = np.arange(0,fNQ-df,df)                                  # Construct frequency axis
     
  elif i==2:
     x=sdeint.itoint(model_MW1, g_fun, z1, t)
     
     y_a = x[:,9]
     y_b = x[:,10]
     y_c = x[:,11]
     
     z_a = x[120001:160000,9]
     z_b = x[120001:160000,10]
     z_c = x[120001:160000,11]
     
     w_a = x[160001:nn1-1,9]
     w_b = x[160001:nn1-1,10]
     w_c = x[160001:nn1-1,11]
     
     dt = Np/(nn1-1)
     N = y_a.shape[0] 
     Tn = N * dt
     
     yf_a = np.fft.fft(y_a - y_a.mean())                              # Compute Fourier transform of x
     S_a = 2 * dt ** 2 / Tn * (yf_a * yf_a.conj())                    # Compute spectrum
     S_a = S_a[:int(len(y_a) / 2)]
     
     yf_b = np.fft.fft(y_b - y_b.mean())                              # Compute Fourier transform of x
     S_b = 2 * dt ** 2 / Tn * (yf_b * yf_b.conj())                    # Compute spectrum
     S_b = S_b[:int(len(y_b) / 2)]
     
     yf_c = np.fft.fft(y_c - y_c.mean())                              # Compute Fourier transform of x
     S_c = 2 * dt ** 2 / Tn * (yf_c * yf_c.conj())                    # Compute spectrum
     S_c = S_c[:int(len(y_c) / 2)]
     
     df = 1 / Tn                                                      # Determine frequency resolution
     fNQ = 1 / dt / 2                                                 # Determine Nyquist frequency
     faxis = np.arange(0,fNQ-df,df)                                   # Construct frequency axis
     
     N = z_a.shape[0] 
     Tn = N * dt
     
     zf_a = np.fft.fft(z_a - z_a.mean())                              # Compute Fourier transform of x
     z_a = 2 * dt ** 2 / Tn * (zf_a * zf_a.conj())                    # Compute spectrum
     z_a = z_a[:int(len(z_a) / 2)]
     
     zf_b = np.fft.fft(z_b - z_b.mean())                              # Compute Fourier transform of x
     z_b = 2 * dt ** 2 / Tn * (zf_b * zf_b.conj())                    # Compute spectrum
     z_b = z_b[:int(len(z_b) / 2)]
     
     zf_c = np.fft.fft(z_c - z_c.mean())                              # Compute Fourier transform of x
     z_c = 2 * dt ** 2 / Tn * (zf_c * zf_c.conj())                    # Compute spectrum
     z_c = z_c[:int(len(z_c) / 2)]
     
     df = 1 / Tn                                                      # Determine frequency resolution
     fNQ = 1 / dt / 2                                                 # Determine Nyquist frequency
     faxisz = np.arange(0,fNQ-df,df)                                  # Construct frequency axis
     
     N = w_a.shape[0] 
     Tn = N * dt
     
     wf_a = np.fft.fft(w_a - w_a.mean())                              # Compute Fourier transform of x
     w_a = 2 * dt ** 2 / Tn * (wf_a * wf_a.conj())                    # Compute spectrum
     w_a = w_a[:int(len(w_a) / 2)]
     
     wf_b = np.fft.fft(w_b - w_b.mean())                              # Compute Fourier transform of x
     w_b = 2 * dt ** 2 / Tn * (wf_b * wf_b.conj())                    # Compute spectrum
     w_b = w_b[:int(len(w_b) / 2)]
     
     wf_c = np.fft.fft(w_c - w_c.mean())                              # Compute Fourier transform of x
     w_c = 2 * dt ** 2 / Tn * (wf_c * wf_c.conj())                    # Compute spectrum
     w_c = w_c[:int(len(w_c) / 2)]
     
     df = 1 / Tn                                                      # Determine frequency resolution
     fNQ = 1 / dt / 2                                                 # Determine Nyquist frequency
     faxisw = np.arange(0,fNQ-df,df)                                  # Construct frequency axis
        

  S_A=np.log(S_a.real)
  S_B=np.log(S_b.real)
  S_C=np.log(S_c.real)
  
  z_A=np.log(z_a.real)
  z_B=np.log(z_b.real)
  z_C=np.log(z_c.real)
  
  w_A=np.log(w_a.real)
  w_B=np.log(w_b.real)
  w_C=np.log(w_c.real)
  
   
  ns=int(nn1/20)
  ns=0
  
  mx=max(x[ns:nn1,0])
  mn=min(x[ns:nn1,0])
     
  axs[0,1].plot(T[ns:nn1],x[ns:nn1,0],color=clr[i])
  axs[0,1].set_ylabel('$s_a$  [nS]')
  axs[0,1].grid()
  axs[0,1].plot([4000,4000],[mn,mx],'k--')
  axs[0,1].plot([8000,8000],[mn,mx],'k--')
  axs[0,1].plot([12000,12000],[mn,mx],'g--')
  axs[0,1].plot([16000,16000],[mn,mx],'g--')

  mx=max(x[ns:nn1,1])
  mn=min(x[ns:nn1,1])
  
  axs[1,1].plot(T[ns:nn1],x[ns:nn1,1],color=clr[i])
  axs[1,1].set_ylabel('$s_b$  [nS]')
  #axs[1,1].set_xlim([0.0,Te])
  axs[1,1].grid()
  axs[1,1].plot([4000,4000],[mn,mx],'k--')
  axs[1,1].plot([8000,8000],[mn,mx],'k--')
  axs[1,1].plot([12000,12000],[mn,mx],'g--')
  axs[1,1].plot([16000,16000],[mn,mx],'g--')
  
  mx=max(x[ns:nn1,2])
  mn=min(x[ns:nn1,2])
  
  axs[2,1].plot(T[ns:nn1],x[ns:nn1,2],color=clr[i])
  axs[2,1].set_ylabel('$s_c$  [nS]')
  #axs[2,1].set_xlim([0.0,Te])
  axs[2,1].grid()
  axs[2,1].set_xlabel('t [ms]')
  axs[2,1].plot([4000,4000],[mn,mx],'k--')
  axs[2,1].plot([8000,8000],[mn,mx],'k--')
  axs[2,1].plot([12000,12000],[mn,mx],'g--')
  axs[2,1].plot([16000,16000],[mn,mx],'g--')
    
  mx=max(x[ns:nn1,9])
  mn=min(x[ns:nn1,9])
  
  axs[0,0].plot(T[ns:nn1],x[ns:nn1,9],color=clr[i])
  axs[0,0].set_ylabel('$r_a$ [Hz]')
  #axs[0,0].set_xlim([0.0,Te])
  axs[0,0].grid()
  axs[0,0].plot([4000,4000],[mn,mx],'k--')
  axs[0,0].plot([8000,8000],[mn,mx],'k--')
  axs[0,0].plot([12000,12000],[mn,mx],'g--')
  axs[0,0].plot([16000,16000],[mn,mx],'g--')
    
  mx=max(x[ns:nn1,10])
  mn=min(x[ns:nn1,10])
  
  axs[1,0].plot(T[ns:nn1],x[ns:nn1,10],color=clr[i])
  #axs[1,0].set_xlim([0.0,Te])
  axs[1,0].set_ylabel('$r_b$ [Hz]')   
  #axs[1,0].set_xlabel('T')
  axs[1,0].grid()
  axs[1,0].plot([4000,4000],[mn,mx],'k--')
  axs[1,0].plot([8000,8000],[mn,mx],'k--')
  axs[1,0].plot([12000,12000],[mn,mx],'g--')
  axs[1,0].plot([16000,16000],[mn,mx],'g--')

  mx=max(x[ns:nn1,11])
  mn=min(x[ns:nn1,11])
  
  axs[2,0].plot(T[ns:nn1],x[ns:nn1,11],color=clr[i])
  #axs[2,0].set_xlim([0.0,Te])
  axs[2,0].set_ylabel('$r_c$ [Hz]')   
  axs[2,0].grid()
  axs[2,0].set_xlabel('t [ms]')
  axs[2,0].plot([4000,4000],[mn,mx],'k--')
  axs[2,0].plot([8000,8000],[mn,mx],'k--')
  axs[2,0].plot([12000,12000],[mn,mx],'g--')
  axs[2,0].plot([16000,16000],[mn,mx],'g--')
  
  print([x[nn1-1,9],x[nn1-1,10],x[nn1-1,11]])
  #print([x[nn1-1,3],x[nn1-1,4],x[nn1-1,5],x[nn1-1,6]])
  
  
  #xx = 3.8
  #zz = 2.2    
  #yy = 3.2
  #I=[x[nn1-1,3],x[nn1-1,4],x[nn1-1,5]]
  #V = Ejac_MW1(I, xx, yy, zz)
  #print(V)

  mxf1=5.0
  mxf2=40.0
  mxf3=5.0
   
  ax[0,0].plot(faxis, S_a.real,color=clr[i])
  ax[0,0].set_ylabel('$r_a$ PSD [V**2/Hz]')   
  ax[0,0].grid()
  #ax[0,0].set_xlabel('frequency [Hz]')
  
  ax[1,0].plot(faxis, S_b.real,color=clr[i])
  ax[1,0].set_ylabel('$r_b$ PSD [V**2/Hz]')   
  ax[1,0].grid()
  #ax[1,0].set_xlabel('frequency [Hz]')
  
  ax[2,0].plot(faxis, S_c.real,color=clr[i])
  ax[2,0].set_ylabel('$r_c$ PSD [V**2/Hz]')   
  ax[2,0].grid()
  ax[2,0].set_xlabel('frequency [Hz]')
  
  #ax[0,0].set_xlim([0.0,mxf1])
  #ax[1,0].set_xlim([0.0,mxf2])
  #ax[2,0].set_xlim([0.0,mxf3])
  
  ax[0,0].set_xlim([5.0,100.0])
  ax[0,0].set_ylim([-0.5,25.0])
  
  ax[1,0].set_xlim([5.0,100.0])
  ax[1,0].set_ylim([-0.5,5.0])
  
  ax[2,0].set_xlim([5.0,100.0])
  ax[2,0].set_ylim([-0.5,36.0])
  
  
   
  ax[0,1].plot(faxis, S_A,color=clr[i])
  ax[0,1].set_ylabel('$r_a$ log(PSD)')   
  ax[0,1].grid()
  #ax[0,1].set_xlabel('frequency [Hz]')
  
  ax[1,1].plot(faxis, S_B,color=clr[i])
  ax[1,1].set_ylabel('$r_b$ log(PSD)')   
  ax[1,1].grid()
  #ax[1,1].set_xlabel('frequency [Hz]')
  
  ax[2,1].plot(faxis, S_C,color=clr[i])
  ax[2,1].set_ylabel('$r_c$ log(PSD)')   
  ax[2,1].grid()
  ax[2,1].set_xlabel('frequency [Hz]')
  
  ax[0,1].set_xlim([0.0,200.0])
  ax[0,1].set_ylim([-15.0,10.0])
  
  ax[1,1].set_xlim([0.0,200.0])
  ax[1,1].set_ylim([-20.0,5.0])
  
  ax[2,1].set_xlim([0.0,200.0])
  ax[2,1].set_ylim([-20.0,10.0])
  
  
  
  ax1[0,0].plot(faxisz, z_a.real,color=clr[i])
  ax1[0,0].set_ylabel('$r_a$ PSD [V**2/Hz]')   
  ax1[0,0].grid()
  #ax1[0,0].set_xlabel('frequency [Hz]')
  
  ax1[1,0].plot(faxisz, z_b.real,color=clr[i])
  ax1[1,0].set_ylabel('$r_b$ PSD [V**2/Hz]')   
  ax1[1,0].grid()
  #ax1[1,0].set_xlabel('frequency [Hz]')
  
  ax1[2,0].plot(faxisz, z_c.real,color=clr[i])
  ax1[2,0].set_ylabel('$r_c$ PSD [V**2/Hz]')   
  ax1[2,0].grid()
  ax1[2,0].set_xlabel('frequency [Hz]')
  
  ax1[0,1].plot(faxisw, w_a.real,color=clr[i])
  ax1[0,1].set_ylabel('$r_a$ PSD [V**2/Hz]')   
  ax1[0,1].grid()
  #ax1[0,1].set_xlabel('frequency [Hz]')
  
  ax1[1,1].plot(faxisw, w_b.real,color=clr[i])
  ax1[1,1].set_ylabel('$r_b$ PSD [V**2/Hz]')   
  ax1[1,1].grid()
  #ax1[1,1].set_xlabel('frequency [Hz]')
  
  ax1[2,1].plot(faxisw, w_c.real,color=clr[i])
  ax1[2,1].set_ylabel('$r_c$ PSD [V**2/Hz]')   
  ax1[2,1].grid()
  ax1[2,1].set_xlabel('frequency [Hz]')

  ax1[0,0].set_xlim([5.0,100.0])
  ax1[1,0].set_xlim([5.0,100.0])
  ax1[2,0].set_xlim([5.0,100.0])
  ax1[0,1].set_xlim([5.0,100.0])
  ax1[1,1].set_xlim([5.0,100.0])
  ax1[2,1].set_xlim([5.0,100.0])
  
  
  ax4.plot3D(x[ns:nn1,1],x[ns:nn1,2],x[ns:nn1,3],color=clr[i])
  ax4.set_xlabel('$s_a$ cnd.. [nS]')  
  ax4.set_ylabel('$s_b$ cnd.. [nS]') 
  ax4.set_zlabel('$s_c$ cnd.. [nS]')
  ax4.grid()





plt.show()
  

teps.toc()    