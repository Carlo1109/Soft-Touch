import numpy as np
from math import cos, sin 


def move_middle(dr, q, links, offset, qmin, qmax):
    q1, q2, q3 = q
    
    # Use a scaling factor > 1 for better numerical accuracy 
    scale = 500
    
    L1, L2, L3 = links

    L1 *= scale
    L2 *= scale
    L3 *= scale

    dr *= scale
    
    # Planar Cartesian Task
    p = np.array([[L1*sin(q1) + L2*sin(q1+q2) + L3*sin(q1+q2+q3)], \
         [L1*cos(q1) + L2*cos(q1+q2) + L3*cos(q1+q2+q3)]])

    # Add offset
    p += np.array([[0], [scale*offset]])
  
    # Position Jacobian: p_dot = Jp(q)q_dot
    Jp = np.array([[L1*cos(q1) + L2*cos(q1+q2) + L3*cos(q1+q2+q3), L2*cos(q1+q2) + L3*cos(q1+q2+q3), L3*cos(q1+q2+q3)], \
         [-L1*sin(q1) - L2*sin(q1+q2) - L3*sin(q1+q2+q3), - L2*sin(q1+q2) - L3*sin(q1+q2+q3), - L3*sin(q1+q2+q3)]])

    # Distance from origin d
    d = np.linalg.norm(p)

    # Weight Matrix for jacobian
    W = [[10.5, 0,  0], 
         [0,  4.5,  0], 
         [0,   0, 7.5]]
    
    # Task jacobian for distance: d_dot = Jd(q)q_dot
    Jd = 1/d*p.transpose().dot(Jp)

    Jd_trans = Jd.transpose()

    W_inv = np.linalg.inv(W)/2

    # Singularity check using weighted determinant
    detJ = np.float64(Jd.dot(W_inv.dot(Jd_trans)))
    # Damped Least Squared
    if detJ <= 1e-5:
        mu = (detJ + 1.0)/1000
    else:
        mu = 0
    
    Jinv = W_inv.dot(Jd_trans).dot(1/(detJ + scale**2*mu**2))
    
    # Proportional Gain
    K = 100

    # Vector for null space control (impose joint limits)
    w = np.array([((qmax + qmin)/2 - q1)/(qmax - qmin), ((qmax + qmin)/2 - q2)/(qmax - qmin), ((qmax + qmin)/2 - q3)/(qmax - qmin)])
    u_null = np.reshape((np.eye(3) - Jinv.dot(Jd)).dot(w.T),[3,1])
    
    # Compute the joint velocities
    qdot = Jinv*K*(dr - d) + u_null

    v1 = qdot[0][0] 
    v2 = qdot[1][0] 
    v3 = qdot[2][0] 
    
    return v1, v2, v3