import numpy as np
from math import pi, cos, sin 
import sympy as sym
from sympy.matrices import Matrix

# Rototranslation using Euler angles and vector x, y, z
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
def symbolic_task_index(positions, orientations, symbols):
    # Approximate small values to zero
    positions = np.where(np.abs(positions) < 1e-6, 0, positions).astype(np.float64)
    orientations = np.where(np.abs(orientations) < 1e-6, 0, orientations).astype(np.float64)

    # Joint1 position
    p0 = positions[0]
    # DH parameters
    r1, r2, r3 = positions[1:, 0]
    d1, d2, d3 = positions[1:, 2]

    o0_1, o1_2, o2_3 = orientations

    TO1 = rotation_euler(o0_1, p0)
    TO2 = rotation_euler(o1_2, [0, 0, 0])
    TO3 = rotation_euler(o2_3, [0, 0, 0])

    j1, j2, j3 = symbols

    # Homogenous transformations from origin to tip according to standard DH convention

    TDH1 = Matrix([[sym.cos(j1), -sym.sin(j1), 0, r1*sym.cos(j1)], \
                   [sym.sin(j1),  sym.cos(j1), 0, r1*sym.sin(j1)], \
                   [0,                      0, 1,             d1], 
                   [0,                      0, 0,              1]])
    
    TDH2 = Matrix([[sym.cos(j2), -sym.sin(j2), 0, r2*sym.cos(j2)], \
                   [sym.sin(j2),  sym.cos(j2), 0, r2*sym.sin(j2)], \
                   [0,                      0, 1,             d2], 
                   [0,                      0, 0,              1]])
    
    TDH3 = Matrix([[sym.cos(j3), -sym.sin(j3), 0, r3*sym.cos(j3)], \
                   [sym.sin(j3),  sym.cos(j3), 0, r3*sym.sin(j3)], \
                   [0,                      0, 1,             d3], 
                   [0,                      0, 0,              1]])
    
    T = Matrix(TO1)*TDH1
    T = T*Matrix(TO2)
    T = sym.simplify(T*Matrix(TDH2))
    T = T*Matrix(TO3)
    T = T*Matrix(TDH3)

    # Cartesian task [x y z]
    pos = (Matrix(T[0:3,3], ndmin=2)).T
    
    # Task Jacobian: p_dot = Jp(q)q_dot
    Jp = Matrix([[sym.diff(pos, j1)], [sym.diff(pos, j2)], [sym.diff(pos, j3)]]).T
    
    return pos, Jp

def actuation_index(dr, q, joint_limits, integral):
    q1, q2, q3 = q

    # Proportional gain
    K = 1.25
    # Integral gain
    K_i = 0

    # Numerical actual position for computational semplicity
    p = np.array([[0.323095324757241*(-0.551087875515956*sin(q1 + q2) + 0.821941607498565*cos(q1 + q2))*sin(q3) + \
                   0.323095324757241*(0.793305710481451*sin(q1 + q2) + 0.569555000795985*cos(q1 + q2))*cos(q3) + 0.377051693903047*sin(q1) + \
                   0.345545925655253*sin(q1 + q2) + 0.0801448116278539*cos(q1) + 0.228954555335644*cos(q1 + q2) + 3.98684038227515e-6, \
                   0.323095324757241*(-0.0716369312968307*sin(q1 + q2) - 0.0480304731098819*cos(q1 + q2) - 0.143376427981922)*sin(q3) + \
                   0.323095324757241*(-0.0496399891300789*sin(q1 + q2) + 0.0691411484230555*cos(q1 + q2) + 0.214293383514247)*cos(q3) - \
                   0.00698508058480545*sin(q1) - 0.0199547043257626*sin(q1 + q2) + 0.0328622204364236*cos(q1) + 0.0301163118039482*cos(q1 + q2) + 0.0834712922729897, \
                   0.323095324757241*(-0.818813871531069*sin(q1 + q2) - 0.548990819771634*cos(q1 + q2) + 0.012543812065275)*sin(q3) + \
                   0.323095324757241*(-0.567387672064595*sin(q1 + q2) + 0.790286942747519*cos(q1 + q2) - 0.0187482417261332)*cos(q3) - 0.079839836423229*sin(q1) - \
                   0.228083314129319*sin(q1 + q2) + 0.375616898372723*cos(q1) + 0.344231019084965*cos(q1 + q2) + 1.8943444491914]]).T

    # Distance of tip from origin
    d = np.linalg.norm(p)

    # Numerical Jacobian for computational semplicity
    Jp = [[0.323095324757241*(-0.821941607498565*sin(q1 + q2) - 0.551087875515956*cos(q1 + q2))*sin(q3) + \
           0.323095324757241*(-0.569555000795985*sin(q1 + q2) + 0.793305710481451*cos(q1 + q2))*cos(q3) - 0.0801448116278539*sin(q1) - \
           0.228954555335644*sin(q1 + q2) + 0.377051693903047*cos(q1) + 0.345545925655253*cos(q1 + q2), \
           0.323095324757241*(-0.821941607498565*sin(q1 + q2) - 0.551087875515956*cos(q1 + q2))*sin(q3) + \
           0.323095324757241*(-0.569555000795985*sin(q1 + q2) + 0.793305710481451*cos(q1 + q2))*cos(q3) - 0.228954555335644*sin(q1 + q2) + \
           0.345545925655253*cos(q1 + q2), 0.323095324757241*(-0.551087875515956*sin(q1 + q2) + 0.821941607498565*cos(q1 + q2))*cos(q3) - \
           0.323095324757241*(0.793305710481451*sin(q1 + q2) + 0.569555000795985*cos(q1 + q2))*sin(q3)], \
          [0.323095324757241*(-0.0691411484230555*sin(q1 + q2) - 0.0496399891300789*cos(q1 + q2))*cos(q3) + 0.323095324757241*(0.0480304731098819*sin(q1 + q2) - \
           0.0716369312968307*cos(q1 + q2))*sin(q3) - 0.0328622204364236*sin(q1) - 0.0301163118039482*sin(q1 + q2) - 0.00698508058480545*cos(q1) - \
           0.0199547043257626*cos(q1 + q2), 0.323095324757241*(-0.0691411484230555*sin(q1 + q2) - 0.0496399891300789*cos(q1 + q2))*cos(q3) + \
           0.323095324757241*(0.0480304731098819*sin(q1 + q2) - 0.0716369312968307*cos(q1 + q2))*sin(q3) - 0.0301163118039482*sin(q1 + q2) - \
           0.0199547043257626*cos(q1 + q2), 0.323095324757241*(-0.0716369312968307*sin(q1 + q2) - 0.0480304731098819*cos(q1 + q2) - 0.143376427981922)*cos(q3) - \
           0.323095324757241*(-0.0496399891300789*sin(q1 + q2) + 0.0691411484230555*cos(q1 + q2) + 0.214293383514247)*sin(q3)], \
          [0.323095324757241*(-0.790286942747519*sin(q1 + q2) - 0.567387672064595*cos(q1 + q2))*cos(q3) + \
           0.323095324757241*(0.548990819771634*sin(q1 + q2) - 0.818813871531069*cos(q1 + q2))*sin(q3) - 0.375616898372723*sin(q1) - \
           0.344231019084965*sin(q1 + q2) - 0.079839836423229*cos(q1) - 0.228083314129319*cos(q1 + q2), \
           0.323095324757241*(-0.790286942747519*sin(q1 + q2) - 0.567387672064595*cos(q1 + q2))*cos(q3) + 0.323095324757241*(0.548990819771634*sin(q1 + q2) - \
           0.818813871531069*cos(q1 + q2))*sin(q3) - 0.344231019084965*sin(q1 + q2) - 0.228083314129319*cos(q1 + q2), \
           0.323095324757241*(-0.818813871531069*sin(q1 + q2) - 0.548990819771634*cos(q1 + q2) + 0.012543812065275)*cos(q3) - \
           0.323095324757241*(-0.567387672064595*sin(q1 + q2) + 0.790286942747519*cos(q1 + q2) - 0.0187482417261332)*sin(q3)]]
    
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
    w = np.array([((joint_limits[0]-pi/4)/2 - q1)/(joint_limits[0] + pi/4),\
                  ((joint_limits[1]/2) - q2)/(joint_limits[1]),\
                  (joint_limits[2]/2 - q3)/(joint_limits[2])])
    u_vinc = (np.eye(3) - Jinv.dot(Jd))*w.transpose()

    integral += dr - d

    # Compute the joint velocities
    qdot = Jinv.dot(K*(dr - d) + K_i*integral) + u_vinc

    return qdot, d, integral