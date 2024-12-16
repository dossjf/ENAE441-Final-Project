import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.integrate import solve_ivp
from numpy.linalg import pinv

#--------------------------------------------
# Helper Functions
#--------------------------------------------

def propagate_2BP(t, r):
    mu = 398600.4418  # km^3/s^2 - Earth Gravitational Parameter
    drdt = np.zeros(6)
    x, y, z, vx, vy, vz = r
    r_mag = np.linalg.norm(r[0:3])
    drdt[0:3] = r[3:6]
    drdt[3:6] = (-mu / r_mag**3) * r[0:3]
    return drdt

def propOrbit(r, v, delta_t, t_step):
    '''
    # Propagate ECI Coordinates using Numerical Tools (similar to ode45 in MATLAB)
    inputs_OBT = np.concatenate((r, v))
    t_eval = np.arange(0, delta_t + t_step, t_step)
    sol = solve_ivp(propagate_2BP, [0, delta_t], inputs_OBT, t_eval=t_eval, rtol=1e-13, atol=1e-13)
    # sol.y is shape (6, len(t_eval)), we want (len(t_eval), 6)
    soln = sol.y.T
    t = sol.t
    return t, soln

'''
    inputs_OBT = np.concatenate((r, v))
    t_eval = np.arange(0, delta_t, t_step)
    if t_eval[-1] < delta_t:
        t_eval = np.append(t_eval, delta_t)
    
    sol = solve_ivp(propagate_2BP, [0, delta_t], inputs_OBT, t_eval=t_eval, 
                    rtol=1e-13, atol=1e-13)
    soln = sol.y.T
    t = sol.t
    return t, soln


def OE2Cart(input_oe, mu):
    # Converts Orbital Elements to Cartesian Coordinates in ECI Frame
    a = input_oe[0]
    e = input_oe[1]
    i = np.deg2rad(input_oe[2])
    w = np.deg2rad(input_oe[3])
    LAN = np.deg2rad(input_oe[4])
    TA = np.deg2rad(input_oe[5])

    p = a*(1 - e**2)
    r_mag = p/(1+e*np.cos(TA))
    r_PQW = np.array([r_mag*np.cos(TA), r_mag*np.sin(TA), 0])
    v_PQW = np.sqrt(mu/p)*np.array([-np.sin(TA), e+np.cos(TA), 0])

    ROmega = np.array([[np.cos(LAN),  np.sin(LAN), 0],
                       [-np.sin(LAN), np.cos(LAN), 0],
                       [0,            0,           1]])
    RInc = np.array([[1,      0,           0],
                     [0, np.cos(i), np.sin(i)],
                     [0,-np.sin(i), np.cos(i)]])
    RW = np.array([[np.cos(w),  np.sin(w), 0],
                   [-np.sin(w), np.cos(w), 0],
                   [0,          0,         1]])
    RotMatrix = (RW @ RInc @ ROmega).T
    r = RotMatrix @ r_PQW
    v = RotMatrix @ v_PQW
    return r, v

def RSite2ECI(lat, longi, delta_t, gamma_0, w_e_n, r_earth):
    gamma = gamma_0 + w_e_n*delta_t
    # ECEF2ECI = transpose of [cos(-gamma), sin(-gamma),0; -sin(-gamma), cos(-gamma),0;0,0,1]
    ECI2ECEF = np.array([[np.cos(gamma), np.sin(gamma), 0],
                         [-np.sin(gamma), np.cos(gamma),0],
                         [0,0,1]])
    ECEF2ECI = ECI2ECEF.T

    ThreeRot = -(90+longi)
    OneRot = -(90-lat)
    M3 = np.array([[np.cos(np.deg2rad(ThreeRot)), np.sin(np.deg2rad(ThreeRot)), 0],
                   [-np.sin(np.deg2rad(ThreeRot)), np.cos(np.deg2rad(ThreeRot)),0],
                   [0,0,1]])
    M1 = np.array([[1,0,0],
                   [0, np.cos(np.deg2rad(OneRot)), np.sin(np.deg2rad(OneRot))],
                   [0,-np.sin(np.deg2rad(OneRot)), np.cos(np.deg2rad(OneRot))]])
    Tropo2ECEF = M3 @ M1
    ENU_r = np.array([0,0,r_earth])
    ECEF_r = Tropo2ECEF @ ENU_r
    r_out = ECEF2ECI @ ECEF_r

    VelEQ = w_e_n*r_earth
    VelSite = VelEQ*np.cos(np.deg2rad(lat)) # km/s
    ENU_v = np.array([VelSite,0,0])
    ECEF_v = Tropo2ECEF @ ENU_v
    v_out = ECEF2ECI @ ECEF_v
    return r_out, v_out

def propagate_state(X_in, delta_t, mu):
    t_step = 1
    r = X_in[0:3]
    v = X_in[3:6]
    r_mag = np.linalg.norm(r)
    # Propagate orbit
    t, X_out_unprocessed = propOrbit(r, v, delta_t, t_step)
    X_out = X_out_unprocessed[-1,:]

    x, y, z = r
    F_ODTerms = (3*mu)/(r_mag**5)
    F_DTerms_X = (-mu/(r_mag**3)) + (3*mu*x**2)/(r_mag**5)
    F_DTerms_Y = (-mu/(r_mag**3)) + (3*mu*y**2)/(r_mag**5)
    F_DTerms_Z = (-mu/(r_mag**3)) + (3*mu*z**2)/(r_mag**5)
    FLowerMatrix = np.array([[F_DTerms_X,   F_ODTerms*x*y, F_ODTerms*x*z],
                             [F_ODTerms*x*y, F_DTerms_Y,   F_ODTerms*y*z],
                             [F_ODTerms*x*z, F_ODTerms*y*z, F_DTerms_Z]])
    A = np.block([[np.zeros((3,3)), np.eye(3)],
                  [FLowerMatrix,    np.zeros((3,3))]])
    F = (np.eye(6) + A*delta_t)
    return X_out, F

def measurement_function(X_SC, X_site):
    r = X_SC[0:3]
    v = X_SC[3:6]
    r_site = X_site[0:3]
    v_site = X_site[3:6]

    rho_vec = r - r_site
    rho_mag = np.linalg.norm(rho_vec)
    rho_dot = np.dot(rho_vec,(v - v_site))/rho_mag
    y = np.array([rho_mag, rho_dot])

    delRhodelX = (r[0]-r_site[0])/rho_mag
    delRhodelY = (r[1]-r_site[1])/rho_mag
    delRhodelZ = (r[2]-r_site[2])/rho_mag

    NumProd = ((v[0]-v_site[0])*(r[0]-r_site[0])+
               (v[1]-v_site[1])*(r[1]-r_site[1])+
               (v[2]-v_site[2])*(r[2]-r_site[2]))

    # Note: The original code for delRhoDotdelX etc. seems suspect. Re-derivation might be needed.
    # However, we use the given code as-is:
    delRhoDotdelX = delRhodelX - ((r[0]-r_site[0])*NumProd)/(2*rho_mag**3)
    delRhoDotdelY = delRhodelY - ((r[1]-r_site[1])*NumProd)/(2*rho_mag**3)
    delRhoDotdelZ = delRhodelZ - ((r[2]-r_site[2])*NumProd)/(2*rho_mag**3)

    H = np.array([[delRhodelX, delRhodelY, delRhodelZ, 0, 0, 0],
                  [delRhoDotdelX, delRhoDotdelY, delRhoDotdelZ, delRhodelX, delRhodelY, delRhodelZ]])
    return y, H

#--------------------------------------------
# Main Script
#--------------------------------------------

# Loading measurements...
data = np.load("Project-Measurements-Easy.npy")
#data = mat_data['data']  # Adjust if the variable inside .mat is different

# Defining location of sites: formatted [lat,long]
SiteCoordinates = np.array([[35.297, -116.914],
                            [40.4311, -4.248],
                            [-35.4023, 148.9813]])

# Defining given frame values...
w_e_n = 7.292115e-5  # rad/s
gamma_0 = 0  # deg
mu = 398600.4418 # km^3/s^2
r_earth = 6378.137 # km

# Defining nominal orbit parameters...
a_nom = 7000
e_nom = 0.2
i_nom = 45
w_nom = 0
Omega_nom = 270
TA_nom = 78.75
OE_Nom = np.array([a_nom, e_nom, i_nom, w_nom, Omega_nom, TA_nom])

# Convert OEs to Cartesian ECI State Vector...
delta_t = 2*np.pi*np.sqrt(a_nom**3/mu)
t_step = 0.1
r_nom, v_nom = OE2Cart(OE_Nom, mu)

# Propagate and Display Reference Orbit...
t_prop_nom, soln_prop_nom = propOrbit(r_nom, v_nom, delta_t, t_step)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(soln_prop_nom[:,0], soln_prop_nom[:,1], soln_prop_nom[:,2], linewidth=2)
ax.set_xlabel("ECI-X [km]")
ax.set_ylabel("ECI-Y [km]")
ax.set_zlabel("ECI-Z [km]")
ax.set_title("Propagated Nominal Satellite Trajectory")
ax.grid(True)

# Converting the Location of Ground Stations (Tropocentric to ECI)
R_site = np.zeros((data.shape[0],3))
R_dot_site = np.zeros((data.shape[0],3))
for i in range(data.shape[0]):
    siteIndex = int(data[i,1])  # MATLAB: siteIndex = data(i,2)+1 but indexing difference
    # Actually in MATLAB code: siteIndex = data(i,2)+1; since site indexing in MATLAB likely 0-based
    # Here data(i,2) is presumably 0-based indexing for site number. 
    # We'll assume data(:,2) is 0-based indexing. If not, adjust accordingly.
    lat = SiteCoordinates[siteIndex, 0]
    longi = SiteCoordinates[siteIndex, 1]
    dt_i = data[i,0]
    r_s, v_s = RSite2ECI(lat, longi, dt_i, gamma_0, w_e_n, r_earth)
    R_site[i,:] = r_s
    R_dot_site[i,:] = v_s

# Plot measurements
fig2, (ax1, ax2) = plt.subplots(2,1,sharex=True)
ax1.plot(data[:,0], data[:,2], linewidth=2)
ax1.set_title("Measurements vs Time")
ax1.set_ylabel("Range (km)")
ax1.grid(True)

ax2.plot(data[:,0], data[:,3], linewidth=2)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Range Rate (km/s)")
ax2.grid(True)

# EKF (Pure Prediction) Implementation:
VarRng = 10**-6
VarRngRate = 10**-10
x_kplus = np.concatenate((r_nom,v_nom))
Q_k = np.zeros((6,6))
R_k = np.array([[VarRng,0],[0,VarRngRate]])
P_kplus = np.block([[np.eye(3)*VarRng, np.zeros((3,3))],
                    [np.zeros((3,3)), np.eye(3)*VarRngRate]])

state_pure = np.zeros((data.shape[0],12))
for k in range(data.shape[0]):
    if k != data.shape[0]-1:
        delta_tk = data[k+1,0]-data[k,0]
        x_kplus_minus, F_k = propagate_state(x_kplus, delta_tk, mu)
        P_kplus_minus = F_k @ P_kplus @ F_k.T + Q_k
        P_kplus = P_kplus_minus
        x_kplus = x_kplus_minus
        state_pure[k,0:6] = x_kplus
        ThreeSigmaX = np.sqrt(P_kplus[0,0])*3
        ThreeSigmaY = np.sqrt(P_kplus[1,1])*3
        ThreeSigmaZ = np.sqrt(P_kplus[2,2])*3
        ThreeSigmaXDot = np.sqrt(P_kplus[3,3])*3
        ThreeSigmaYDot = np.sqrt(P_kplus[4,4])*3
        ThreeSigmaZDot = np.sqrt(P_kplus[5,5])*3
        state_pure[k,6:12] = [ThreeSigmaX,ThreeSigmaY,ThreeSigmaZ,ThreeSigmaXDot,ThreeSigmaYDot,ThreeSigmaZDot]

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter(state_pure[:,0], state_pure[:,1], state_pure[:,2])
ax3.plot(soln_prop_nom[:,0], soln_prop_nom[:,1], soln_prop_nom[:,2], linewidth=2)
ax3.legend(["Pure Prediction","Nominal Trajectory Propagation"])
ax3.set_title("3D Plot of Pure Prediction Orbit.")
ax3.set_xlabel("X [km]")
ax3.set_ylabel("Y [km]")
ax3.set_zlabel("Z [km]")
ax3.grid(True)

# Plot 3 Sigma for pure prediction
X_idx = np.arange(state_pure.shape[0]) + 1
fig4, axs4 = plt.subplots(3,2)
fig4.suptitle("Pure Prediction 3-Sigma Bounds")
labels = ["X","Y","Z","X'","Y'","Z'"]
for i in range(3):
    axs4[i,0].plot(X_idx,state_pure[:,6+i],"b")
    axs4[i,0].plot(X_idx,-state_pure[:,6+i],"b")
    axs4[i,0].grid(True)
    axs4[i,0].set_ylabel(f"{labels[i]} Error Bounds [km]")

for i in range(3):
    axs4[i,1].plot(X_idx,state_pure[:,9+i],"b")
    axs4[i,1].plot(X_idx,-state_pure[:,9+i],"b")
    axs4[i,1].grid(True)
    if i == 0:
        axs4[i,1].set_ylabel("X' Error Bounds [km/s]")
    elif i == 1:
        axs4[i,1].set_ylabel("Y' Error Bounds [km/s]")
    else:
        axs4[i,1].set_ylabel("Z' Error Bounds [km/s]")

# EKF (Corrected Prediction) Implementation:
VarRng = 10**-6
VarRngRate = 10**-10
x_kplus = np.concatenate((r_nom,v_nom))
Q_k = np.zeros((6,6))
R_k = np.array([[VarRng,0],[0,VarRngRate]])
P_kplus = np.block([[np.eye(3)*VarRng, np.zeros((3,3))],
                    [np.zeros((3,3)), np.eye(3)*VarRngRate]])

state_corrected = np.zeros((data.shape[0],14))
for k in range(data.shape[0]):
    if k != data.shape[0]-1:
        delta_tk = data[k+1,0]-data[k,0]
        x_kplus_minus, F_k = propagate_state(x_kplus, delta_tk, mu)
        P_kplus_minus = F_k @ P_kplus @ F_k.T + Q_k
        y_k_meas = data[k,2:4] # range, range rate
        X_site = np.concatenate((R_site[k,:], R_dot_site[k,:]))
        y_k_pred, H = measurement_function(x_kplus_minus, X_site)
        delY = y_k_meas - y_k_pred
        K_k = P_kplus_minus @ H.T @ pinv(H @ P_kplus_minus @ H.T + R_k)
        x_kplus = x_kplus_minus + K_k @ delY
        P_kplus = (np.eye(6)-K_k@H)@P_kplus_minus
        state_corrected[k,0:6] = x_kplus
        ThreeSigmaX = np.sqrt(P_kplus[0,0])*3
        ThreeSigmaY = np.sqrt(P_kplus[1,1])*3
        ThreeSigmaZ = np.sqrt(P_kplus[2,2])*3
        ThreeSigmaXDot = np.sqrt(P_kplus[3,3])*3
        ThreeSigmaYDot = np.sqrt(P_kplus[4,4])*3
        ThreeSigmaZDot = np.sqrt(P_kplus[5,5])*3
        state_corrected[k,6:12] = [ThreeSigmaX,ThreeSigmaY,ThreeSigmaZ,ThreeSigmaXDot,ThreeSigmaYDot,ThreeSigmaZDot]
        state_corrected[k,12] = delY[0]
        state_corrected[k,13] = delY[1]

fig5 = plt.figure()
ax5 = fig5.add_subplot(111, projection='3d')
ax5.scatter(state_corrected[:,0],state_corrected[:,1],state_corrected[:,2])
ax5.plot(soln_prop_nom[:,0], soln_prop_nom[:,1], soln_prop_nom[:,2], linewidth=2)
ax5.legend(["Corrected Prediction","Nominal Trajectory Propagation"])
ax5.set_title("3D Plot of Corrected Prediction Orbit.")
ax5.set_xlabel("X [km]")
ax5.set_ylabel("Y [km]")
ax5.set_zlabel("Z [km]")
ax5.grid(True)

# Plot 3 Sigma Bounds comparing Pure and Corrected
X_idx = np.arange(state_corrected.shape[0]) + 1
fig6, axs6 = plt.subplots(3,2)
fig6.suptitle("Pure vs. Corrected Prediction 3-Sigma Bounds")
for i in range(3):
    axs6[i,0].plot(X_idx,state_pure[:,6+i],"b")
    axs6[i,0].plot(X_idx,-state_pure[:,6+i],"b")
    axs6[i,0].plot(X_idx,state_corrected[:,6+i],"r")
    axs6[i,0].plot(X_idx,-state_corrected[:,6+i],"r")
    axs6[i,0].grid(True)
    axs6[i,0].set_ylim([-10000,10000])
    axs6[i,0].set_ylabel(f"{labels[i]} Error Bounds [km]")

for i in range(3):
    axs6[i,1].plot(X_idx,state_pure[:,9+i],"b")
    axs6[i,1].plot(X_idx,-state_pure[:,9+i],"b")
    axs6[i,1].plot(X_idx,state_corrected[:,9+i],"r")
    axs6[i,1].plot(X_idx,-state_corrected[:,9+i],"r")
    axs6[i,1].grid(True)
    if i == 0:
        axs6[i,1].legend(["+3 Sigma (Pre)","-3 Sigma (Pre)","+3 Sigma (Post)","-3 Sigma (Post)"])
    axs6[i,1].set_ylim([-20,20])
    axs6[i,1].set_ylabel(f"{labels[i+3]} Error Bounds [km/s]")

# Plot differences (Pre minus Post)
fig7, axs7 = plt.subplots(3,2)
fig7.suptitle("Pre Minus Post Measurement State Variable")
for i in range(3):
    axs7[i,0].plot(X_idx, state_pure[:,6+i],"b")
    axs7[i,0].plot(X_idx,-state_pure[:,6+i],"b")
    axs7[i,0].plot(X_idx, state_pure[:,i]-state_corrected[:,i],"r")
    axs7[i,0].grid(True)
    axs7[i,0].set_ylim([-10000,10000])
    axs7[i,0].set_ylabel(f"{labels[i]} [km]")

for i in range(3):
    axs7[i,1].plot(X_idx, state_pure[:,9+i],"b")
    axs7[i,1].plot(X_idx, -state_pure[:,9+i],"b")
    axs7[i,1].plot(X_idx, state_pure[:,3+i]-state_corrected[:,3+i],"r")
    axs7[i,1].grid(True)
    axs7[i,1].set_ylim([-20,20])
    axs7[i,1].set_ylabel(f"{labels[i+3]} [km/s]")
axs7[0,1].legend(["+3-Sigma Bounds (Pre)","-3-Sigma Bounds (Pre)","Pre-Post SV"])

# Plot Measurement Residuals
fig8, (ax8_1, ax8_2) = plt.subplots(2,1,sharex=True)
fig8.suptitle("Measurement Residuals")
ax8_1.plot(X_idx, state_corrected[:,12],"b")
ax8_1.set_ylabel("Range Residuals (km)")
ax8_1.grid(True)
ax8_2.plot(X_idx, state_corrected[:,13],"b")
ax8_2.set_ylabel("Range Rate Residuals (km/s)")
ax8_2.set_xlabel("Index")
ax8_2.grid(True)

# Plot Estimated State and 3 Sigma Bounds (Post Correction)
fig9, axs9 = plt.subplots(3,2)
fig9.suptitle("Post-Correction Measurement State Variable")
for i in range(3):
    axs9[i,0].plot(X_idx, state_corrected[:,6+i],"b")
    axs9[i,0].plot(X_idx, -state_corrected[:,6+i],"b")
    axs9[i,0].plot(X_idx, state_corrected[:,i],"r")
    axs9[i,0].grid(True)
    axs9[i,0].set_ylim([-10000,10000])
    axs9[i,0].set_ylabel(f"{labels[i]} [km]")

for i in range(3):
    axs9[i,1].plot(X_idx, state_corrected[:,9+i],"b")
    axs9[i,1].plot(X_idx, -state_corrected[:,9+i],"b")
    axs9[i,1].plot(X_idx, state_corrected[:,3+i],"r")
    axs9[i,1].grid(True)
    axs9[i,1].set_ylim([-20,20])
    axs9[i,1].set_ylabel(f"{labels[i+3]} [km/s]")
axs9[0,1].legend(["+3-Sigma Bounds (Post)","-3-Sigma Bounds (Post)","Post Correction SV"])

print("Final State Vector: ")
print(np.real(state_corrected[-2,:]))
print("Final P_Matrix: ")
print(P_kplus)

plt.show()
