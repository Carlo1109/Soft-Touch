import numpy as np
from math import cos, sin 
import sympy as sym
from sympy.matrices import Matrix 

def rotation_euler(euler_angles, translation):
        alpha, beta, gamma = euler_angles
        x, y, z = translation
        
        Rx =  np.array([[1,          0,           0],
                        [0, cos(alpha), -sin(alpha)],
                        [0, sin(alpha),  cos(alpha)]])
        
        Ry =  np.array([[cos(beta),  0, sin(beta)],
                        [0,          1,         0],
                        [-sin(beta), 0, cos(beta)]])
        
        Rz =  np.array([[cos(gamma), -sin(gamma), 0],
                        [sin(gamma),  cos(gamma), 0],
                        [         0,           0, 1]])
        
        R = Rx.dot(Ry)
        R = R.dot(Rz)
        T = np.column_stack((R, np.array([x, y, z])))
        T = np.row_stack((T, np.array([0, 0, 0, 1])))
        # Approximate small values to zero
        return np.where(np.abs(T) < 1e-6, 0, T)

# Calculation of symbolic task and Jacobian
def symbolic_task_thumb(positions, orientations, symbols):
    # Approximate small values to zero
    positions = np.where(np.abs(positions) < 1e-6, 0, positions).astype(np.float64)
    orientations = np.where(np.abs(orientations) < 1e-6, 0, orientations).astype(np.float64)

    # Joint1 position
    p0 = positions[0]
    # DH parameters
    r1, r2 = positions[1:, 0]
    d1, d2 = positions[1:, 2]

    o0_1, o1_2 = orientations

    TO1 = rotation_euler(o0_1, p0)
    TO2 = rotation_euler(o1_2, [0, 0, 0])

    j1, j2 = symbols

    # Homogenous transformations from origin to tip according to standard DH convention

    TDH1 = Matrix([[sym.cos(j1), -sym.sin(j1), 0, r1*sym.cos(j1)], \
                   [sym.sin(j1),  sym.cos(j1), 0, r1*sym.sin(j1)], \
                   [0,                      0, 1,             d1], 
                   [0,                      0, 0,              1]])
    
    TDH2 = Matrix([[sym.cos(j2), -sym.sin(j2), 0, r2*sym.cos(j2)], \
                   [sym.sin(j2),  sym.cos(j2), 0, r2*sym.sin(j2)], \
                   [0,                      0, 1,             d2], 
                   [0,                      0, 0,              1]])
    
    T = Matrix(TO1)*TDH1
    T = T*Matrix(TO2)
    T = sym.simplify(T*Matrix(TDH2))

    # Cartesian task [x y z]
    pos = (Matrix(T[0:3,3], ndmin=2)).T
    
    # Task Jacobian: p_dot = Jp(q)q_dot
    Jp = Matrix([[sym.diff(pos, j1)], [sym.diff(pos, j2)]]).T
    
    return pos, Jp

def actuation_thumb(dr, q, joint_limits):
    q1, q2 = q

    # Proportional gain
    K = 1.0

    # Numerical actual position for computational semplicity
    p = np.array([[-0.371409328296358*(0.83356811462125*sin(q1) - 0.389116168235393*cos(q1))*sin(q2) + 0.371409328296358*(0.382690473263435*sin(q1) + 0.819802933731448*cos(q1) - 0.0709653839844241)*cos(q2) + 0.315576937063046*sin(q1) + 0.123473076437244*cos(q1) + 0.205225178322721, 0.371409328296358*(0.0548707341534264*sin(q1) + 0.764625384036207*cos(q1))*sin(q2) + 0.371409328296358*(0.751998693380009*sin(q1) - 0.0539646227416178*cos(q1) + 0.116214763743811)*cos(q2) + 0.190007109634082*sin(q1) - 0.208907170936981*cos(q1) - 0.371309409948572, -0.371409328296358*(0.549684819528645*sin(q1) + 0.513747632311126*cos(q1))*sin(q2) + 0.371409328296358*(-0.505263827608875*sin(q1) + 0.540607563764575*cos(q1) + 0.119216082397095)*cos(q2) + 0.00262890000440023*sin(q1) + 0.277147270163453*cos(q1) + 1.37880097862452]]).T
    
    # Distance of tip from origin
    d = np.linalg.norm(p)

    # Numerical Jacobian for computational semplicity
    Jp = [[0.371409328296358*(-0.819802933731448*sin(q1) + 0.382690473263435*cos(q1))*cos(q2) - 0.371409328296358*(0.389116168235393*sin(q1) + 0.83356811462125*cos(q1))*sin(q2) - 0.123473076437244*sin(q1) + 0.315576937063046*cos(q1), -0.371409328296358*(0.83356811462125*sin(q1) - 0.389116168235393*cos(q1))*cos(q2) - 0.371409328296358*(0.382690473263435*sin(q1) + 0.819802933731448*cos(q1) - 0.0709653839844241)*sin(q2)], [0.371409328296358*(-0.764625384036207*sin(q1) + 0.0548707341534264*cos(q1))*sin(q2) + 0.371409328296358*(0.0539646227416178*sin(q1) + 0.751998693380009*cos(q1))*cos(q2) + 0.208907170936981*sin(q1) + 0.190007109634082*cos(q1), 0.371409328296358*(0.0548707341534264*sin(q1) + 0.764625384036207*cos(q1))*cos(q2) - 0.371409328296358*(0.751998693380009*sin(q1) - 0.0539646227416178*cos(q1) + 0.116214763743811)*sin(q2)], [0.371409328296358*(-0.540607563764575*sin(q1) - 0.505263827608875*cos(q1))*cos(q2) - 0.371409328296358*(-0.513747632311126*sin(q1) + 0.549684819528645*cos(q1))*sin(q2) - 0.277147270163453*sin(q1) + 0.00262890000440023*cos(q1), -0.371409328296358*(0.549684819528645*sin(q1) + 0.513747632311126*cos(q1))*cos(q2) - 0.371409328296358*(-0.505263827608875*sin(q1) + 0.540607563764575*cos(q1) + 0.119216082397095)*sin(q2)]]
    
    # Extended jacobian for distance: d_dot = Jd(q)q_dot
    Jd = 1/d * p.T.dot(Jp)
    
    # Jacobian Damped Least Squared (DLS)
    Jd_trans = Jd.transpose()
    detJ = np.float64(Jd.dot(Jd_trans))
    if detJ <= 1e-1:
        mu = (detJ + 1.0)/5
    else:
        mu = 0
    Jinv = Jd_trans.dot(1/(detJ + mu**2))

    # Vector for null space control
    w = np.array([((joint_limits[0])/2 - q1)/(joint_limits[0]), (joint_limits[1]/2 - q2)/(joint_limits[1])])
    u_vinc = (np.eye(2) - Jinv.dot(Jd))*w.transpose()
    
    # Compute the joint velocities
    qdot = Jinv.dot(K*(dr - d)) + u_vinc

    return qdot, d