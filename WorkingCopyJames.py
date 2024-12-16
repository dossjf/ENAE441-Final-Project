import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

def propagate_2BP(t, r):
    mu = 398600.4418  # km^3/s^2 - Earth Gravitational Parameter
    drdt = np.zeros(6)
    x, y, z, vx, vy, vz = r
    r_mag = np.linalg.norm(r[0:3])
    drdt[0:3] = r[3:6]
    drdt[3:6] = (-mu / r_mag**3) * r[0:3]
    return drdt

def propOrbit(r, v, delta_t, t_step): #Propagates ECI Coordinates using Numerical Tools
    inputs_OBT = np.concatenate((r, v))
    t_eval = np.arange(0, delta_t, t_step)
    if t_eval[-1] < delta_t:
        t_eval = np.append(t_eval, delta_t)
    
    sol = solve_ivp(propagate_2BP, [0, delta_t], inputs_OBT, t_eval=t_eval, 
                    rtol=1e-13, atol=1e-13) #Change parameter to change propgator tolerance.
    soln = sol.y.T
    t = sol.t
    return t, soln

def OE2Cart(input_oe, mu): #Converts Orbital Elements to Cartesian Coordinates in ECI Frame
    #Process inputs and convert degrees to radians...
    a = input_oe[0]
    e = input_oe[1]
    i = np.deg2rad(input_oe[2])
    w = np.deg2rad(input_oe[3])
    LAN = np.deg2rad(input_oe[4])
    TA = np.deg2rad(input_oe[5])

    #Calculate quantities used in conversion...
    p = a*(1 - e**2) #Semi Latus Rectum
    r_mag = p/(1+e*np.cos(TA)) #Magnitude of R
    r_PQW = np.array([r_mag*np.cos(TA), r_mag*np.sin(TA), 0]) #Perifocal Position
    v_PQW = np.sqrt(mu/p)*np.array([-np.sin(TA), e+np.cos(TA), 0]) #Perifocal Velocity

    #Define 3-1-3 Rotation Matrix...
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

    #Execute Transformation...
    r = RotMatrix @ r_PQW
    v = RotMatrix @ v_PQW
    return r, v

def RSite2ECI(lat, long, delta_t, gamma_0, w_e_n, r_earth): #Converts site position to ECI state-vec at a given T past 0.
    gamma = gamma_0 + w_e_n*delta_t
    ECI2ECEF = np.array([[np.cos(-gamma), np.sin(-gamma), 0],
                         [-np.sin(-gamma), np.cos(-gamma),0],
                         [0,0,1]])
    ECEF2ECI = ECI2ECEF.T #Transposing the ECI2ECEF matrix to invert it.

    ThreeRot = -(90+long) #Expects long and lat in degrees
    OneRot = -(90-lat)
    M3 = np.array([[np.cos(np.deg2rad(ThreeRot)), np.sin(np.deg2rad(ThreeRot)), 0],
                   [-np.sin(np.deg2rad(ThreeRot)), np.cos(np.deg2rad(ThreeRot)),0],
                   [0,0,1]])
    M1 = np.array([[1,0,0],
                   [0, np.cos(np.deg2rad(OneRot)), np.sin(np.deg2rad(OneRot))],
                   [0,-np.sin(np.deg2rad(OneRot)), np.cos(np.deg2rad(OneRot))]])
    Tropo2ECEF = M3 @ M1
    ENU_r = np.array([0,0,r_earth]) #A position of 0,0,r_earth relative to the center of the earth is the site. 
    ECEF_r = Tropo2ECEF @ ENU_r.T
    r_out = (ECEF2ECI @ ECEF_r).T
    VelEQ = w_e_n*r_earth #Tangential Velocity of the Earth at the Equator.
    VelSite = VelEQ*np.cos(np.deg2rad(lat)) #Tangential Velocity of the site. Km/s;
    ENU_v = np.array([VelSite,0,0]) #Earth's rotational velocity is only eastward.
    ECEF_v = Tropo2ECEF @ ENU_v.T
    v_out = (ECEF2ECI @ ECEF_v).T
    return r_out, v_out

def propagate_state(X_in, delta_t, mu):
    t_step = 1 #This is the single biggest determinent on how long the kalman filter takes to run. Start with 1 second.
    r = X_in[0:3]
    v = X_in[3:6]
    r_mag = np.linalg.norm(r)
    # Propagate orbit
    t, X_out_unprocessed = propOrbit(r, v, delta_t, t_step)
    X_out = X_out_unprocessed[-1,:]

    x = r[0]
    y = r[1]
    z = r[2]
    F_ODTerms = (3*mu)/(r_mag**5) #Multiply by relevant x,y terms.
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

    delRhodelX = (2*r[0] - 2*r_site[0])/(2*((r[0] - r_site[0])**2 + (r[1] - r_site[1])**2 + (r[2] - r_site[2])**2)**(1/2))
    delRhodelY = (2*r[1] - 2*r_site[1])/(2*((r[0] - r_site[0])**2 + (r[1] - r_site[1])**2 + (r[2] - r_site[2])**2)**(1/2))
    delRhodelZ = (2*r[2] - 2*r_site[2])/(2*((r[0] - r_site[0])**2 + (r[1] - r_site[1])**2 + (r[2] - r_site[2])**2)**(1/2))

    delRhoDotdelX = (2*v[0] - 2*v_site[0])/(2*((r[0] - r_site[0])**2 + (r[1] - r_site[1])**2 + (r[2] - r_site[2])**2)**(1/2)) - ((2*r[0] - 2*r_site[0])*(2*(v[0] - v_site[0])*(r[0] - r_site[0]) + 2*(v[1] - v_site[1])*(r[1] - r_site[1]) + 2*(v[2] - v_site[2])*(r[2] - r_site[2])))/(4*((r[0] - r_site[0])**2 + (r[1] - r_site[1])**2 + (r[2] - r_site[2])**2)**(3/2))
    delRhoDotdelY = (2*v[1] - 2*v_site[1])/(2*((r[0] - r_site[0])**2 + (r[1] - r_site[1])**2 + (r[2] - r_site[2])**2)**(1/2)) - ((2*r[1] - 2*r_site[1])*(2*(v[0] - v_site[0])*(r[0] - r_site[0]) + 2*(v[1] - v_site[1])*(r[1] - r_site[1]) + 2*(v[2] - v_site[2])*(r[2] - r_site[2])))/(4*((r[0] - r_site[0])**2 + (r[1] - r_site[1])**2 + (r[2] - r_site[2])**2)**(3/2))
    delRhoDotdelZ = (2*v[2] - 2*v_site[2])/(2*((r[0] - r_site[0])**2 + (r[1] - r_site[1])**2 + (r[2] - r_site[2])**2)**(1/2)) - ((2*r[2] - 2*r_site[2])*(2*(v[0] - v_site[0])*(r[0] - r_site[0]) + 2*(v[1] - v_site[1])*(r[1] - r_site[1]) + 2*(v[2] - v_site[2])*(r[2] - r_site[2])))/(4*((r[0] - r_site[0])**2 + (r[1] - r_site[1])**2 + (r[2] - r_site[2])**2)**(3/2))

    H = np.array([[delRhodelX, delRhodelY, delRhodelZ, 0, 0, 0],
                  [delRhoDotdelX, delRhoDotdelY, delRhoDotdelZ, delRhodelX, delRhodelY, delRhodelZ]])
    return y, H

if __name__ == "__main__":
    #Project Step 0a: - Defining Initial Values and Givens
    #Loading measurements...
    current_directory = os.path.dirname(os.path.abspath(__file__))
    filename = 'Project-Measurements-Easy.npy'  #Note, this expects the measurements file to be in the same rootdir as this .py file.
    file_path = os.path.join(current_directory, filename)
    data = np.load(file_path, allow_pickle=True)

    #Defining location of sites: formatted [lat,long]
    SiteCoordinates = np.array([[35.297, -116.914],
                                [40.4311, -4.248],
                                [-35.4023, 148.9813]])

    #Defining given frame values...
    w_e_n = 7.292115e-5  #rad/s - Rotation Rate of Earth.
    gamma_0 = 0  #rad - Local Sidereal Time.
    mu = 398600.4418 #km^3/s^2 - Earth Gravitational Parameter.
    r_earth = 6378.137 #km - Radius of Earth.
    PositionalErrors = 10**3 #The Positional 

    #Defining nominal orbit parameters...
    a_nom = 7000 #km - Semi-Major Axis of Nominal Orbit.
    e_nom = 0.2 #dimless - Eccentricity of the Nominal Orbit
    i_nom = 45 #deg - Inclination of the Nominal Orbit
    w_nom = 0 #deg - Argument of Periapsis of the Nominal Orbit
    Omega_nom = 270 #deg - Longitude of Ascending Node of the Nominal Orbit
    TA_nom = 78.75 #deg - True Anomaly of the Nominal Orbit at T=0
    OE_Nom = np.array([a_nom, e_nom, i_nom, w_nom, Omega_nom, TA_nom]) #mixed - Array of Nominal Orbital Elements

    #Project Step 0b: - Inspecting the Trajectory of the Nominal Orbit
    #Convert OEs to Cartesian ECI State Vector...
    delta_t = 2*np.pi*np.sqrt(a_nom**3/mu) #Propagate nominal trajectory for one orbit.
    t_step = 0.1
    r_nom, v_nom = OE2Cart(OE_Nom, mu)

    #Propagate and Display Reference Orbit...
    t_prop_nom, soln_prop_nom = propOrbit(r_nom, v_nom, delta_t, t_step)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(soln_prop_nom[:,0], soln_prop_nom[:,1], soln_prop_nom[:,2], linewidth=2)
    ax.set_xlabel("ECI-X [km]")
    ax.set_ylabel("ECI-Y [km]")
    ax.set_zlabel("ECI-Z [km]")
    ax.set_title("Propagated Nominal Satellite Trajectory")
    ax.grid(True)
    plt.savefig("NominalOrbit.png")

    #Project Step 0c: - Converting the Location of Ground Stations (Tropocentric to ECI)
    #Define ECI Position Array of Active Ground Station...
    R_site = np.zeros((data.shape[0],3)) #Columns are X,Y,Z ECI Coords of Active Site, each row corresponds to the corresponding active time in the data.
    R_dot_site = np.zeros((data.shape[0],3))
    for i in range(data.shape[0]):
        siteIndex = int(data[i,1])
        siteLat = SiteCoordinates[siteIndex, 0]
        siteLong = SiteCoordinates[siteIndex, 1]
        delta_t = data[i,0]
        r_site, v_site = RSite2ECI(siteLat, siteLong, delta_t, gamma_0, w_e_n, r_earth)
        R_site[i,:] = r_site
        R_dot_site[i,:] = v_site
    #Project Step 1: - Problem Setup
    #d.) Plot the measurements as a function of time.
    fig2, (ax1, ax2) = plt.subplots(2,1,sharex=True)
    ax1.plot(data[:,0], data[:,2], linewidth=2)
    ax1.set_title("Measurements vs Time")
    ax1.set_ylabel("Range (km)")
    ax1.grid(True)
    ax2.plot(data[:,0], data[:,3], linewidth=2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Range Rate (km/s)")
    ax2.grid(True)
    plt.savefig("Measurements.png")

    #EKF (Pure Prediction) Implementation:
    #Initialize
    VarRng = 10**-6 #Range Variance - km^2
    VarRngRate = 10**-10 #Range Rate Variance - km^2/s^2
    x_kplus = np.concatenate((r_nom,v_nom)) #x_kplus = x_0;
    Q_k = np.zeros((6,6)) #Define Q_0
    R_k = np.array([[VarRng,0],[0,VarRngRate]]) #Define R_0
    P_kplus = np.block([[np.eye(3)*VarRng, np.zeros((3,3))],
                        [np.zeros((3,3)), np.eye(3)*VarRngRate]]) #Define P_0
    #Create Storage
    state_pure = np.zeros((data.shape[0],12)) #Formatted [X,Y,Z,X',Y',Z',3sigmaX,3sigmaY,3sigmaZ,3sigmaX',3sigmaY',3sigmaZ']
    for k in range(data.shape[0]):
        if k != data.shape[0]-1:
            #Find Delta_T
            delta_tk = data[k+1,0]-data[k,0]
            #Predict
            x_kplus_minus, F_k = propagate_state(x_kplus, delta_tk, mu)
            P_kplus_minus = F_k @ P_kplus @ F_k.T + Q_k
            P_kplus = P_kplus_minus
            x_kplus = x_kplus_minus
            #Store
            state_pure[k,0:6] = x_kplus
            ThreeSigmaX = np.sqrt(P_kplus[0,0])*3
            ThreeSigmaY = np.sqrt(P_kplus[1,1])*3
            ThreeSigmaZ = np.sqrt(P_kplus[2,2])*3
            ThreeSigmaXDot = np.sqrt(P_kplus[3,3])*3
            ThreeSigmaYDot = np.sqrt(P_kplus[4,4])*3
            ThreeSigmaZDot = np.sqrt(P_kplus[5,5])*3
            state_pure[k,6:12] = [ThreeSigmaX,ThreeSigmaY,ThreeSigmaZ,ThreeSigmaXDot,ThreeSigmaYDot,ThreeSigmaZDot]
    #Plot 3D Trace
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.scatter(state_pure[:,0], state_pure[:,1], state_pure[:,2],color='blue')
    ax3.plot(soln_prop_nom[:,0], soln_prop_nom[:,1], soln_prop_nom[:,2], linewidth=2,color='red')
    ax3.legend(["Pure Prediction","Nominal Trajectory Propagation"])
    ax3.set_title("3D Plot of Pure Prediction Orbit.")
    ax3.set_xlabel("X [km]")
    ax3.set_ylabel("Y [km]")
    ax3.set_zlabel("Z [km]")
    ax3.grid(True)
    plt.savefig("PurePredOrbit.png")

    #Plot 3 Sigma Stuff
    X = np.arange(1, state_pure.shape[0] + 1)
    fig4, axs4 = plt.subplots(3, 2, figsize=(10, 8))
    fig4.suptitle("Pure Prediction 3-Sigma Bounds")
    axs4[0, 0].fill_between(X, state_pure[:, 6], -state_pure[:, 6], color="blue", alpha=0.3)
    axs4[0, 0].grid(True)
    axs4[0, 0].set_ylabel("X Error Bounds [km]")
    axs4[1, 0].fill_between(X, state_pure[:, 7], -state_pure[:, 7], color="blue", alpha=0.3)
    axs4[1, 0].grid(True)
    axs4[1, 0].set_ylabel("Y Error Bounds [km]")
    axs4[2, 0].fill_between(X, state_pure[:, 8], -state_pure[:, 8], color="blue", alpha=0.3)
    axs4[2, 0].grid(True)
    axs4[2, 0].set_ylabel("Z Error Bounds [km]")
    axs4[0, 1].fill_between(X, state_pure[:, 9], -state_pure[:, 9], color="blue", alpha=0.3)
    axs4[0, 1].grid(True)
    axs4[0, 1].set_ylabel("X' Error Bounds [km/s]")
    axs4[1, 1].fill_between(X, state_pure[:, 10], -state_pure[:, 10], color="blue", alpha=0.3)
    axs4[1, 1].grid(True)
    axs4[1, 1].set_ylabel("Y' Error Bounds [km/s]")
    axs4[2, 1].fill_between(X, state_pure[:, 11], -state_pure[:, 11], color="blue", alpha=0.3)
    axs4[2, 1].grid(True)
    axs4[2, 1].set_ylabel("Z' Error Bounds [km/s]")    
    plt.savefig("PurePredThreeSigma.png")
    #EKF (Corrected Prediction) Implementation:
    #Initialize
    VarRng = 10**-6 #Range Variance - km^2
    VarRngRate = 10**-10 #Range Rate Variance - km^2/s^2
    x_kplus = np.concatenate((r_nom,v_nom)) #x_kplus = x_0;
    Q_k = np.zeros((6,6)) #Define Q_0
    R_k = np.array([[VarRng,0],[0,VarRngRate]]) #Define R_0
    P_kplus = np.block([[np.eye(3)*VarRng, np.zeros((3,3))],
                        [np.zeros((3,3)), np.eye(3)*VarRngRate]]) #Define P_0
    #Create Storage
    state_corrected = np.zeros((data.shape[0],14)) #Same format as before but the last two columns are for range and range rate residuals respectively.
    for k in range(data.shape[0]):
        if k != data.shape[0]-1:
            #Find Delta_T
            delta_tk = data[k+1,0]-data[k,0]
            #Predict
            x_kplus_minus, F_k = propagate_state(x_kplus, delta_tk, mu)
            P_kplus_minus = F_k @ P_kplus @ F_k.T + Q_k
            #Correct
            y_k_meas = data[k,2:4] # range, range rate
            X_site = np.concatenate((R_site[k,:], R_dot_site[k,:]))
            y_k_pred, H = measurement_function(x_kplus_minus, X_site)
            delY = y_k_meas - y_k_pred
            K_k = P_kplus_minus @ H.T @ np.linalg.pinv(H @ P_kplus_minus @ H.T + R_k)
            x_kplus = x_kplus_minus + K_k @ delY
            P_kplus = (np.eye(6)-K_k@H)@P_kplus_minus
            #Store
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
    #Plot 3D Trace
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111, projection='3d')
    ax5.scatter(state_corrected[:,0],state_corrected[:,1],state_corrected[:,2],color='blue')
    ax5.plot(soln_prop_nom[:,0], soln_prop_nom[:,1], soln_prop_nom[:,2], linewidth=2,color='red')
    ax5.legend(["Corrected Prediction","Nominal Trajectory Propagation"])
    ax5.set_title("3D Plot of Corrected Prediction Orbit.")
    ax5.set_xlabel("X [km]")
    ax5.set_ylabel("Y [km]")
    ax5.set_zlabel("Z [km]")
    ax5.grid(True)
    plt.savefig("CorrectedThreeD.png")

    #Plot 3 Sigma Bounds comparing Pure and Corrected
    X = np.arange(1, state_pure.shape[0] + 1)
    fig6, axs6 = plt.subplots(3, 2, figsize=(10, 8))
    fig6.suptitle("Pure vs. Corrected Prediction 3-Sigma Bounds")
    axs6[0, 0].fill_between(X, state_pure[:, 6], -state_pure[:, 6], color="blue", alpha=0.3)
    axs6[0, 0].fill_between(X, state_corrected[:, 6], -state_corrected[:, 6], color="red", alpha=0.3)
    axs6[0, 0].legend(["Pure Prediction","Corrected"])
    axs6[0, 0].grid(True)
    axs6[0, 0].set_ylabel("X Error Bounds [km]")
    axs6[1, 0].fill_between(X, state_pure[:, 7], -state_pure[:, 7], color="blue", alpha=0.3)
    axs6[1, 0].fill_between(X, state_corrected[:, 7], -state_corrected[:, 7], color="red", alpha=0.3)
    axs6[1, 0].grid(True)
    axs6[1, 0].set_ylabel("Y Error Bounds [km]")
    axs6[2, 0].fill_between(X, state_pure[:, 8], -state_pure[:, 8], color="blue", alpha=0.3)
    axs6[2, 0].fill_between(X, state_corrected[:, 8], -state_corrected[:, 8], color="red", alpha=0.3)
    axs6[2, 0].grid(True)
    axs6[2, 0].set_ylabel("Z Error Bounds [km]")
    axs6[0, 1].fill_between(X, state_pure[:, 9], -state_pure[:, 9], color="blue", alpha=0.3)
    axs6[0, 1].fill_between(X, state_corrected[:, 9], -state_corrected[:, 9], color="red", alpha=0.3)
    axs6[0, 1].grid(True)
    axs6[0, 1].set_ylabel("X' Error Bounds [km/s]")
    axs6[1, 1].fill_between(X, state_pure[:, 10], -state_pure[:, 10], color="blue", alpha=0.3)
    axs6[1, 1].fill_between(X, state_corrected[:, 10], -state_corrected[:, 10], color="red", alpha=0.3)
    axs6[1, 1].grid(True)
    axs6[1, 1].set_ylabel("Y' Error Bounds [km/s]")
    axs6[2, 1].fill_between(X, state_pure[:, 11], -state_pure[:, 11], color="blue", alpha=0.3)
    axs6[2, 1].fill_between(X, state_corrected[:, 11], -state_corrected[:, 11], color="red", alpha=0.3)
    axs6[2, 1].grid(True)
    axs6[2, 1].set_ylabel("Z' Error Bounds [km/s]")    
    plt.savefig("PurevsCorrectedThreeSigma.png")

    #Plot differences (Pre minus Post)
    X = np.arange(1, state_pure.shape[0] + 1)
    fig7, axs7 = plt.subplots(3, 2, figsize=(10, 8))
    fig7.suptitle("Pre Minus Post Measurement State Variable")
    axs7[0, 0].fill_between(X, state_pure[:, 6], -state_pure[:, 6], color="blue", alpha=0.3)
    axs7[0, 0].plot(X, state_pure[:, 0]-state_corrected[:, 0], color="red", alpha=0.6,linewidth=2)
    axs7[0, 0].legend(["Pure Prediction Bounds","Pre Minus Post Measurement SV"])
    axs7[0, 0].grid(True)
    axs7[0, 0].set_ylabel("X [km]")
    axs7[1, 0].fill_between(X, state_pure[:, 7], -state_pure[:, 7], color="blue", alpha=0.3)
    axs7[1, 0].plot(X, state_pure[:, 1]-state_corrected[:, 1], color="red", alpha=0.6,linewidth=2)
    axs7[1, 0].grid(True)
    axs7[1, 0].set_ylabel("Y [km]")
    axs7[2, 0].fill_between(X, state_pure[:, 8], -state_pure[:, 8], color="blue", alpha=0.3)
    axs7[2, 0].plot(X, state_pure[:, 2]-state_corrected[:, 2], color="red", alpha=0.6,linewidth=2)
    axs7[2, 0].grid(True)
    axs7[2, 0].set_ylabel("Z [km]")
    axs7[0, 1].fill_between(X, state_pure[:, 9], -state_pure[:, 9], color="blue", alpha=0.3)
    axs7[0, 1].plot(X, state_pure[:, 3]-state_corrected[:, 3], color="red", alpha=0.6,linewidth=2)
    axs7[0, 1].grid(True)
    axs7[0, 1].set_ylabel("X' [km/s]")
    axs7[1, 1].fill_between(X, state_pure[:, 10], -state_pure[:, 10], color="blue", alpha=0.3)
    axs7[1, 1].plot(X, state_pure[:, 4]-state_corrected[:, 4], color="red", alpha=0.6,linewidth=2)
    axs7[1, 1].grid(True)
    axs7[1, 1].set_ylabel("Y' [km/s]")
    axs7[2, 1].fill_between(X, state_pure[:, 11], -state_pure[:, 11], color="blue", alpha=0.3)
    axs7[2, 1].plot(X, state_pure[:, 5]-state_corrected[:, 5], color="red", alpha=0.6,linewidth=2)
    axs7[2, 1].grid(True)
    axs7[2, 1].set_ylabel("Z' [km/s]")    
    plt.savefig("PureMinusCorrectedSV.png")

    #Plot Measurement Residuals
    fig8, (ax8_1, ax8_2) = plt.subplots(2,1,sharex=True)
    fig8.suptitle("Measurement Residuals")
    ax8_1.plot(X, state_corrected[:,12],linewidth=2)
    ax8_1.set_ylabel("Range Residuals (km)")
    ax8_1.grid(True)
    ax8_2.plot(X, state_corrected[:,13],linewidth=2)
    ax8_2.set_ylabel("Range Rate Residuals (km/s)")
    ax8_2.set_xlabel("Index")
    ax8_2.grid(True)
    plt.savefig("Residuals.png")

    #Plot Estimated State and 3 Sigma Bounds (Post Correction)
    X = np.arange(1, state_pure.shape[0] + 1)
    fig9, axs9 = plt.subplots(3, 2, figsize=(10, 8))
    fig9.suptitle("Post-Correction State Variable")
    axs9[0, 0].fill_between(X, state_corrected[:, 6], -state_corrected[:, 6], color="blue", alpha=0.3)
    axs9[0, 0].plot(X, state_corrected[:, 0], color="red", alpha=0.6,linewidth=2)
    axs9[0, 0].legend(["3-Sigma Bounds","Best Estimate SV"])
    axs9[0, 0].grid(True)
    axs9[0, 0].set_ylabel("X [km]")
    axs9[1, 0].fill_between(X, state_corrected[:, 7], -state_corrected[:, 7], color="blue", alpha=0.3)
    axs9[1, 0].plot(X, state_corrected[:, 1], color="red", alpha=0.6,linewidth=2)
    axs9[1, 0].grid(True)
    axs9[1, 0].set_ylabel("Y [km]")
    axs9[2, 0].fill_between(X, state_corrected[:, 8], -state_corrected[:, 8], color="blue", alpha=0.3)
    axs9[2, 0].plot(X, state_corrected[:, 2], color="red", alpha=0.6,linewidth=2)
    axs9[2, 0].grid(True)
    axs9[2, 0].set_ylabel("Z [km]")
    axs9[0, 1].fill_between(X, state_corrected[:, 9], -state_corrected[:, 9], color="blue", alpha=0.3)
    axs9[0, 1].plot(X, state_corrected[:, 3], color="red", alpha=0.6,linewidth=2)
    axs9[0, 1].grid(True)
    axs9[0, 1].set_ylabel("X' [km/s]")
    axs9[1, 1].fill_between(X, state_corrected[:, 10], -state_corrected[:, 10], color="blue", alpha=0.3)
    axs9[1, 1].plot(X, state_corrected[:, 4], color="red", alpha=0.6,linewidth=2)
    axs9[1, 1].grid(True)
    axs9[1, 1].set_ylabel("Y' [km/s]")
    axs9[2, 1].fill_between(X, state_corrected[:, 11], -state_corrected[:, 11], color="blue", alpha=0.3)
    axs9[2, 1].plot(X, state_corrected[:, 5], color="red", alpha=0.6,linewidth=2)
    axs9[2, 1].grid(True)
    axs9[2, 1].set_ylabel("Z' [km/s]")    
    plt.savefig("CorrectedSVs.png")

    print("Final State Vector: ")
    print(np.real(state_corrected[-2,:]))
    print("Final P_Matrix: ")
    print(P_kplus)

    plt.show()
