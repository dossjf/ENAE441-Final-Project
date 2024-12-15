%Project Step 0a: - Defining Initial Values and Givens
    %Loading measurements...
    load("Project-Measurements-Easy.mat")

    %Defining location of sites - formatted [lat,long] with row index corresponding to site number...
    SiteCoordinates = [35.297,-116.914;40.4311,-4.248;-35.4023,148.9813];

    %Defining given frame values...
    w_e_n = 7.292115*10^-5; %rad/s - Rotation Rate of Earth.
    gamma_0 = 0; %deg - Local Sidereal Time.
    mu = 398600.4418; %km^3/s^2 - Earth Gravitational Parameter.
    r_earth = 6378.137; %km - Radius of Earth.

    %Defining nominal orbit parameters...
    a_nom = 7000; %km - Semi-Major Axis of Nominal Orbit.
    e_nom = 0.2; %dimless - Eccentricity of the Nominal Orbit
    i_nom = 45; %deg - Inclination of the Nominal Orbit
    w_nom = 0; %deg - Argument of Periapsis of the Nominal Orbit
    Omega_nom = 270; %deg - Longitude of Ascending Node of the Nominal Orbit
    TA_nom = 78.75; %deg - True Anomaly of the Nominal Orbit at T=0
    OE_Nom = [a_nom,e_nom,i_nom,w_nom,Omega_nom,TA_nom]; %mixed - Array of Nominal Orbital Elements

%Project Step 0b: - Inspecting the Trajectory of the Nominal Orbit
    %Convert OEs to Cartesian ECI State Vector...
    delta_t = 2*pi*sqrt(a_nom^3/mu); %Propagate nominal trajectory for one orbit.
    t_step = 0.1;
    [r_nom,v_nom] = OE2Cart(OE_Nom,mu);

    %Propagate and Display Reference Orbit...
    [t_prop_nom,soln_prop_nom] = propOrbit(r_nom,v_nom,delta_t,t_step);
    figure()
    hold on %Will overlay kalman-filter orbital estimates onto this later.
    plot3(soln_prop_nom(:,1),soln_prop_nom(:,2),soln_prop_nom(:,3),LineWidth=2)
    axis equal
    grid on
    xlabel("ECI-X [km]")
    ylabel("ECI-Y [km]")
    zlabel("ECI-Z [km]")
    title("Propagated Nominal Satellite Trajectory")

%Project Step 0c: - Converting the Location of Ground Stations (Tropocentric to ECI)
    %Define ECI Position Array of Active Ground Station...
    R_site = zeros(size(data,1),3); %Columns are X,Y,Z ECI Coords of Active Site, each row corresponds to the corresponding active time in the data.
    R_dot_site = zeros(size(data,1),3);
    for i = 1:size(data,1)
        siteIndex = data(i,2)+1;
        siteLat = SiteCoordinates(siteIndex,1);
        siteLong = SiteCoordinates(siteIndex,2);
        delta_t = data(i,1);
        [r_site,v_site] = RSite2ECI(siteLat,siteLong,delta_t,gamma_0,w_e_n,r_earth);
        R_site(i,:) = r_site;
        R_dot_site(i,:) = v_site;
    end

%Project Step 1: - Problem Setup
    %a.) Express the non-linear system in continous time SS form.
    %b.) Convert to discrete time form.
    %c.) Modify to include stochasticity.
    %d.) Plot the measurements as a function of time.
    figure()
    subplot(2,1,1)
    plot(data(:,1),data(:,3),"LineWidth",2);
    title("Measurements vs Time")
    xlabel("Time (s)")
    ylabel("Range (km)")
    subplot(2,1,2)
    plot(data(:,1),data(:,4),"LineWidth",2);
    xlabel("Time (s)")
    ylabel("Range Rate (km/s)")

%EKF (Pure Prediction) Implementation:
    %Initialize
    VarRng = 10^-6; %Range Variance - km^2
    VarRngRate = 10^-10; %Range Rate Variance - km^2/s^2
    x_kplus = [r_nom;v_nom]; %x_kplus = x_0;
    Q_k = zeros(6,6); %Define Q_0
    R_k = [VarRng,0;VarRngRate,0]; %Define R_0
    P_kplus = [eye(3,3)*VarRng,zeros(3,3);zeros(3,3),eye(3,3)*VarRngRate]; %Define P_0;
    %Create Storage
    state_pure = zeros(size(data,1),12); %Formatted [X,Y,Z,X',Y',Z',3sigmaX,3sigmaY,3sigmaZ,3sigmaX',3sigmaY',3sigmaZ']
    for k = 1:size(data,1)
        %Find Delta_T
        if(k ~= size(data,1))
            delta_t = data(k+1,1)-data(k,1);
            %Predict
            [x_kplus_minus,F_k] = propagate_state(x_kplus,delta_t,mu);
            P_kplus_minus = F_k*P_kplus*F_k'+Q_k;
            P_kplus = P_kplus_minus;
            x_kplus = x_kplus_minus;
            %Store
            state_pure(k,(1:6)) = x_kplus;
            ThreeSigmaX = sqrt(P_kplus(1,1))*3;
            ThreeSigmaY = sqrt(P_kplus(2,2))*3;
            ThreeSigmaZ = sqrt(P_kplus(3,3))*3;
            ThreeSigmaXDot = sqrt(P_kplus(4,4))*3;
            ThreeSigmaYDot = sqrt(P_kplus(5,5))*3;
            ThreeSigmaZDot = sqrt(P_kplus(6,6))*3;
            state_pure(k,7:12) = [ThreeSigmaX,ThreeSigmaY,ThreeSigmaZ,ThreeSigmaXDot,ThreeSigmaYDot,ThreeSigmaZDot];
            %Correct
                %Not Implemented Here
            %Store
                %Not Implemented Here
        end
    end
    %Plot 3D Trace
        figure()
        hold on
        axis equal
        scatter3(state_pure(:,1),state_pure(:,2),state_pure(:,3))
        plot3(soln_prop_nom(:,1),soln_prop_nom(:,2),soln_prop_nom(:,3),LineWidth=2)
        legend(["Pure Prediction","Nominal Trajectory Propagation"])
        title("3D Plot of Pure Prediction Orbit.")
        xlabel("X [km]")
        ylabel("Y [km]")
        zlabel("Z [km]")
        grid on
    %Plot 3 Sigma Stuff
        X = (1:size(state_pure,1));
        figure()
        sgtitle("Pure Prediction 3-Sigma Bounds")
        subplot(3,2,1)
        hold on
        plot(X,state_pure(:,7),"Color","blue")
        plot(X,-state_pure(:,7),"Color","blue")
        grid on
        ylabel("X Error Bounds [km]")
        subplot(3,2,3)
        hold on
        plot(X,state_pure(:,8),"Color","blue")
        plot(X,-state_pure(:,8),"Color","blue")
        grid on
        ylabel("Y Error Bounds [km]")
        subplot(3,2,5)
        hold on
        plot(X,state_pure(:,9),"Color","blue")
        plot(X,-state_pure(:,9),"Color","blue")
        grid on
        ylabel("Z Error Bounds [km]")
        subplot(3,2,2)
        hold on
        plot(X,state_pure(:,10),"Color","blue")
        plot(X,-state_pure(:,10),"Color","blue")
        grid on
        ylabel("X' Error Bounds [km/s]")
        subplot(3,2,4)
        hold on
        plot(X,state_pure(:,11),"Color","blue")
        plot(X,-state_pure(:,11),"Color","blue")
        grid on
        ylabel("Y' Error Bounds [km/s]")
        subplot(3,2,6)
        hold on
        plot(X,state_pure(:,12),"Color","blue")
        plot(X,-state_pure(:,12),"Color","blue")
        grid on
        ylabel("Z' Error Bounds [km/s]")

%EKF (Corrected Prediction) Implementation:
    %Initialize
    VarRng = 10^-6; %Range Variance - km^2
    VarRngRate = 10^-10; %Range Rate Variance - km^2/s^2
    x_kplus = [r_nom;v_nom]; %x_kplus = x_0;
    Q_k = zeros(6,6); %Define Q_0
    R_k = [VarRng,0;VarRngRate,0]; %Define R_0
    P_kplus = [eye(3,3)*VarRng,zeros(3,3);zeros(3,3),eye(3,3)*VarRngRate]; %Define P_0;
    %Create Storage
    state_corrected = zeros(size(data,1),14); %Same format as before but the last two columns are for range and range rate residuals respectively.
    for k = 1:size(data,1)
        %Find Delta_T
        if(k ~= size(data,1))
            delta_t = data(k+1,1)-data(k,1);
            %Predict
            [x_kplus_minus,F_k] = propagate_state(x_kplus,delta_t,mu);
            P_kplus_minus = F_k*P_kplus*F_k'+Q_k;
            %Correct
            y_k_meas = data(k,(3:4))';
            X_site = [R_site(k,:)';R_dot_site(k,:)'];
            [y_k_pred,H] = measurement_function(x_kplus_minus,X_site);
            delY = y_k_meas-y_k_pred;
            K_k = P_kplus_minus*H'*pinv(H*P_kplus_minus*H'+R_k);
            x_kplus = x_kplus_minus + K_k*delY;
            P_kplus = (eye(6,6)-K_k*H)*P_kplus_minus;
            %Store
            state_corrected(k,(1:6)) = x_kplus;
            ThreeSigmaX = sqrt(P_kplus(1,1))*3;
            ThreeSigmaY = sqrt(P_kplus(2,2))*3;
            ThreeSigmaZ = sqrt(P_kplus(3,3))*3;
            ThreeSigmaXDot = sqrt(P_kplus(4,4))*3;
            ThreeSigmaYDot = sqrt(P_kplus(5,5))*3;
            ThreeSigmaZDot = sqrt(P_kplus(6,6))*3;
            state_corrected(k,7:12) = [ThreeSigmaX,ThreeSigmaY,ThreeSigmaZ,ThreeSigmaXDot,ThreeSigmaYDot,ThreeSigmaZDot];
            state_corrected(k,13) = delY(1);
            state_corrected(k,13) = delY(2);
        end
    end
    %Plot 3D Trace
        figure()
        hold on
        axis equal
        scatter3(state_corrected(:,1),state_corrected(:,2),state_corrected(:,3))
        plot3(soln_prop_nom(:,1),soln_prop_nom(:,2),soln_prop_nom(:,3),LineWidth=2)
        legend(["Corrected Prediction","Nominal Trajectory Propagation"])
        title("3D Plot of Corrected Prediction Orbit.")
        xlabel("X [km]")
        ylabel("Y [km]")
        zlabel("Z [km]")
        grid on
    %Plot 3 Sigma Stuff
        X = (1:size(state_corrected,1));
        figure()
        sgtitle("Pure vs. Corrected Prediction 3-Sigma Bounds")
        subplot(3,2,1)
        hold on
        plot(X,state_pure(:,7),"Color","blue")
        plot(X,-state_pure(:,7),"Color","blue")
        plot(X,state_corrected(:,7),"Color","red")
        plot(X,-state_corrected(:,7),"Color","red")
        grid on
        ylim([-10000,10000])
        ylabel("X Error Bounds [km]")
        subplot(3,2,3)
        hold on
        plot(X,state_pure(:,8),"Color","blue")
        plot(X,-state_pure(:,8),"Color","blue")
        plot(X,state_corrected(:,8),"Color","red")
        plot(X,-state_corrected(:,8),"Color","red")
        grid on
        ylim([-10000,10000])
        ylabel("Y Error Bounds [km]")
        subplot(3,2,5)
        hold on
        plot(X,state_pure(:,9),"Color","blue")
        plot(X,-state_pure(:,9),"Color","blue")
        plot(X,state_corrected(:,9),"Color","red")
        plot(X,-state_corrected(:,9),"Color","red")
        grid on
        ylim([-10000,10000])
        ylabel("Z Error Bounds [km]")
        subplot(3,2,2)
        hold on
        plot(X,state_pure(:,10),"Color","blue")
        plot(X,-state_pure(:,10),"Color","blue")
        plot(X,state_corrected(:,10),"Color","red")
        plot(X,-state_corrected(:,10),"Color","red")
        legend(["+3 Sigma (Pre)","-3 Sigma (Pre)","+3 Sigma (Post)","-3 Sigma (Post)"])
        grid on
        ylim([-20,20])
        ylabel("X' Error Bounds [km/s]")
        subplot(3,2,4)
        hold on
        plot(X,state_pure(:,11),"Color","blue")
        plot(X,-state_pure(:,11),"Color","blue")
        plot(X,state_corrected(:,11),"Color","red")
        plot(X,-state_corrected(:,11),"Color","red")
        grid on
        ylim([-20,20])
        ylabel("Y' Error Bounds [km/s]")
        subplot(3,2,6)
        hold on
        plot(X,state_pure(:,12),"Color","blue")
        plot(X,-state_pure(:,12),"Color","blue")
        plot(X,state_corrected(:,12),"Color","red")
        plot(X,-state_corrected(:,12),"Color","red")
        grid on
        ylim([-20,20])
        ylabel("Z' Error Bounds [km/s]")
%Plot 3 Sigma Stuff Continued
        X = (1:size(state_corrected,1));
        figure()
        sgtitle("Pre Minus Post Measurement State Variable")
        subplot(3,2,1)
        hold on
        plot(X,state_pure(:,7),"Color","blue")
        plot(X,-state_pure(:,7),"Color","blue")
        plot(X,state_pure(:,1)-state_corrected(:,1),"Color","red")
        ylim([-10000,10000])
        grid on
        ylabel("X [km]")
        subplot(3,2,3)
        hold on
        plot(X,state_pure(:,8),"Color","blue")
        plot(X,-state_pure(:,8),"Color","blue")
        plot(X,state_pure(:,2)-state_corrected(:,2),"Color","red")
        ylim([-10000,10000])
        grid on
        ylabel("Y [km]")
        subplot(3,2,5)
        hold on
        plot(X,state_pure(:,9),"Color","blue")
        plot(X,-state_pure(:,9),"Color","blue")
        plot(X,state_pure(:,3)-state_corrected(:,3),"Color","red")
        ylim([-10000,10000])
        grid on
        ylabel("Z [km]")
        subplot(3,2,2)
        hold on
        plot(X,state_pure(:,10),"Color","blue")
        plot(X,-state_pure(:,10),"Color","blue")
        plot(X,state_pure(:,4)-state_corrected(:,4),"Color","red")
        legend(["+3-Sigma Bounds (Pre)","-3-Sigma Bounds (Pre)","Pre-Post SV"])
        grid on
        ylabel("X' [km/s]")
        ylim([-20,20])
        subplot(3,2,4)
        hold on
        plot(X,state_pure(:,11),"Color","blue")
        plot(X,-state_pure(:,11),"Color","blue")
        plot(X,state_pure(:,5)-state_corrected(:,5),"Color","red")
        grid on
        ylabel("Y' [km/s]")
        ylim([-20,20])
        subplot(3,2,6)
        hold on
        plot(X,state_pure(:,12),"Color","blue")
        plot(X,-state_pure(:,12),"Color","blue")
        plot(X,state_pure(:,6)-state_corrected(:,6),"Color","red")
        grid on
        ylabel("Z' [km/s]")
        ylim([-20,20])
%Plot Residuals
        figure()
        sgtitle("Measurement Residuals")
        subplot(2,1,1)
        plot(X,state_corrected(:,13),"Color","blue")
        ylabel("Range Residuals (km)")
        grid on
        xlim([0,size(X,2)])
        subplot(2,1,2)
        plot(X,state_corrected(:,14),"Color","blue")
        ylabel("Range Rate Residuals (km/s)")
        xlim([0,size(X,2)])
        grid on
%Plot Estimated State and 3 Sigma Bounds
        X = (1:size(state_corrected,1));
        figure()
        sgtitle("Post-Correction Measurement State Variable")
        subplot(3,2,1)
        hold on
        plot(X,state_corrected(:,7),"Color","blue")
        plot(X,-state_corrected(:,7),"Color","blue")
        plot(X,state_corrected(:,1),"Color","red")
        ylim([-10000,10000])
        grid on
        ylabel("X [km]")
        subplot(3,2,3)
        hold on
        plot(X,state_corrected(:,8),"Color","blue")
        plot(X,-state_corrected(:,8),"Color","blue")
        plot(X,state_corrected(:,2),"Color","red")
        ylim([-10000,10000])
        grid on
        ylabel("Y [km]")
        subplot(3,2,5)
        hold on
        plot(X,state_corrected(:,9),"Color","blue")
        plot(X,-state_corrected(:,9),"Color","blue")
        plot(X,state_corrected(:,3),"Color","red")
        ylim([-10000,10000])
        grid on
        ylabel("Z [km]")
        subplot(3,2,2)
        hold on
        plot(X,state_corrected(:,10),"Color","blue")
        plot(X,-state_corrected(:,10),"Color","blue")
        plot(X,state_corrected(:,4),"Color","red")
        legend(["+3-Sigma Bounds (Post)","-3-Sigma Bounds (Post)","Post Correction SV"])
        grid on
        ylabel("X' [km/s]")
        ylim([-20,20])
        subplot(3,2,4)
        hold on
        plot(X,state_corrected(:,11),"Color","blue")
        plot(X,-state_corrected(:,11),"Color","blue")
        plot(X,state_corrected(:,5),"Color","red")
        grid on
        ylabel("Y' [km/s]")
        ylim([-20,20])
        subplot(3,2,6)
        hold on
        plot(X,state_corrected(:,12),"Color","blue")
        plot(X,-state_corrected(:,12),"Color","blue")
        plot(X,state_corrected(:,6),"Color","red")
        grid on
        ylabel("Z' [km/s]")
        ylim([-20,20])
%Final State and Uncertainty
    disp("Final State Vector: ")
    disp(real(state_corrected(end-1,:)))
    disp("Final P_Matrix: ")
    disp(P_kplus)
%Project Addendum: Helper Functions

function [X_out, F] = propagate_state(X_in, delta_t, mu)
    t_step = 1; %This is the single biggest determinent on how long the kalman filter takes to run. Start with 1 second.
    r = X_in(1:3);
    v = X_in(4:6);
    r_mag = norm(r);
    [t,X_out_unprocessed] = propOrbit(r,v,delta_t,t_step);
    X_out = X_out_unprocessed(end,:)';

    x = r(1);
    y = r(2);
    z = r(3);
    F_ODTerms = (3*mu)/(r_mag^5); %Multiply by relevant x,y terms.
    F_DTerms_X = (-mu/(r_mag^3)) + (3*mu*x^2)/(r_mag^5);
    F_DTerms_Y = (-mu/(r_mag^3)) + (3*mu*y^2)/(r_mag^5);
    F_DTerms_Z = (-mu/(r_mag^3)) + (3*mu*z^2)/(r_mag^5);
    FLowerMatrix = [F_DTerms_X, F_ODTerms*x*y, F_ODTerms*x*z;
                    F_ODTerms*x*y, F_DTerms_Y, F_ODTerms*y*z;
                    F_ODTerms*x*z, F_ODTerms*y*z, F_DTerms_Z];
    A = [zeros(3,3),eye(3,3);FLowerMatrix,zeros(3,3)];
    F = (eye(6,6)+A*delta_t);
end

function [y,H] = measurement_function(X_SC,X_site)
    r = X_SC(1:3);
    v = X_SC(4:6);
    r_site = X_site(1:3);
    v_site = X_site(4:6);
    rho_vec = r-r_site;
    rho_mag = norm(rho_vec);
    rho_dot = dot(rho_vec,(v-v_site))/rho_mag;
    y = [rho_mag;rho_dot];

    delRhodelX = (r(1)-r_site(1))/rho_mag;
    delRhodelY = (r(2)-r_site(2))/rho_mag;
    delRhodelZ = (r(3)-r_site(3))/rho_mag;
    NumProd = (v(1)-v_site(1))*(r(1)-r_site(1))+(v(2)-v_site(2))*(r(2)-r_site(2))+(v(3)-v_site(3))*(r(3)-r_site(3));
    delRhoDotdelX = delRhodelX - ((r(1)-r_site(1))*(NumProd))/(2*rho_mag^3);
    delRhoDotdelY = delRhodelY - ((r(2)-r_site(2))*(NumProd))/(2*rho_mag^3);
    delRhoDotdelZ = delRhodelZ - ((r(3)-r_site(3))*(NumProd))/(2*rho_mag^3);
    H = [delRhodelX delRhodelY delRhodelZ 0 0 0;
         delRhoDotdelX delRhoDotdelY delRhoDotdelZ delRhodelX delRhodelY delRhodelZ];
end
function drdt = propagate_2BP(t,r) %Orbital Dynamics Diff-EQ Function
    mu = 398600.4418; %km^3/s^2 - Earth Gravitational Parameter.
    drdt = zeros(6, 1);
    drdt(1:3) = r(4:6);
    r_mag = norm(r(1:3));
    drdt(4:6) = (-mu / r_mag^3) * r(1:3);
end

function [t,soln] = propOrbit(r,v,delta_t,t_step) %Propagates ECI Coordinates using Numerical Tools
    inputs_OBT = [r;v];
    t_range = 0:t_step:delta_t;
    tolerance = 10^-13; %Change parameter to change propgator tolerance.
    options = odeset('RelTol',tolerance,'AbsTol',tolerance);
    [t, soln] = ode45(@propagate_2BP, t_range, inputs_OBT, options);
end

function [r,v] = OE2Cart(input,mu) %Converts Orbital Elements to Cartesian Coordinates in ECI Frame
        %Process inputs and convert degrees to radians...
        a = input(1);
        e = input(2);
        i = deg2rad(input(3));
        w = deg2rad(input(4));
        LAN = deg2rad(input(5));
        TA = deg2rad(input(6));

        %Calculate quantities used in conversion...
        p = a*(1-e^2); %Semi Latus Rectum
        r_mag = p/(1+e*cos(TA)); %Magnitude of R
        r_PQW = [r_mag * cos(TA); r_mag * sin(TA); 0]; %Perifocal Position
        v_PQW = sqrt(mu / p) * [-sin(TA); (e + cos(TA)); 0]; %Perifocal Velocity
        
        %Define 3-1-3 Rotation Matrix...
        ROmega = [cos(LAN),  sin(LAN), 0; 
                  -sin(LAN), cos(LAN), 0; 
                  0,         0,        1];
        RInc = [1,  0,          0; 
                0,  cos(i),     sin(i);
                0, -sin(i),     cos(i)];
        RW = [cos(w),  sin(w), 0; 
              -sin(w), cos(w), 0;
              0,       0,      1];
        RotMatrix = (RW * RInc * ROmega)';

        %Execute Transformation...
        r = RotMatrix * r_PQW;
        v = RotMatrix * v_PQW;
end

function [r,v] = RSite2ECI(lat,long,delta_t,gamma_0,w_e_n,r_earth) %Converts site position to ECI state-vec at a given T past 0.
    gamma = gamma_0 + w_e_n*delta_t;
    ECEF2ECI = transpose([cos(-gamma),  sin(-gamma), 0; 
                          -sin(-gamma), cos(-gamma), 0;
                          0,       0,      1]); %Transposing the ECI2ECEF matrix to invert it.
    ThreeRot = -(90+long); %Expects long and lat in degrees
    OneRot = -(90-lat);
    M3 = [cosd(ThreeRot),  sind(ThreeRot), 0; 
          -sind(ThreeRot), cosd(ThreeRot), 0;
          0,       0,      1];
    M1 =[1,  0,          0; 
         0,  cosd(OneRot),     sind(OneRot);
        0 , -sind(OneRot),     cosd(OneRot)];
    Tropo2ECEF = M3*M1;
    ENU_r = [0,0,r_earth]; %A position of 0,0,r_earth relative to the center of the earth is the site. 
    ECEF_r = Tropo2ECEF*ENU_r';
    r = transpose(ECEF2ECI*ECEF_r);
    VelEQ = w_e_n*r_earth; %Tangential Velocity of the Earth at the Equator.
    VelSite = VelEQ*cosd(lat); %Tangential Velocity of the site. Km/s;
    ENU_v = [VelSite,0,0]; %Earth's rotational velocity is only eastward.
    ECEF_v = Tropo2ECEF*ENU_v';
    v = transpose(ECEF2ECI*ECEF_v);
end
