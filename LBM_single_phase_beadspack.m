%% D2Q9 BGK LBM simulating single phase gravity driven flow through a 2D
%% cross section of a glass beads pack
%%
%% Author: Rui Xu, The University of Texas at Austin, 2017
%% This code is for your own use, please do not distribute.

%% clear the workspace
clear
close all
clc

tic

%% Initialization
%% Set parameters
tau = 1.0;               % relaxation time 
g = 0.00001;             % gravity or other force
density = 1;          
tf = 10001;            % maximum number of iteration steps
precision = 1e-5;      % covergence criterion
vold = 1000;
statIter = 100;        % check if convergence is achived every 100 time steps
data = imread('beads_dry.slice150.sub.tif'); % data is 0s and 255s

data = data/255; %get 0s and 1s from original data

% I will require fluid nodes to be 0, and 1 will be solid.
is_solid_node = data;    % define geometry

[ny,nx] = size(data);

% allocate memory for distribution function
f = zeros(ny,nx,9);   % particle distribution f1, f2, ... f9 for every point of the image. 
                      % for point data(i,j) related distributions are
                      % stored as f(i,j,1:9)
feq = f;              % equilibrium function feq
ftemp = feq;


% initialize distribution function
f(:,:,1) = (4./9. )*density;
f(:,:,2) = (1./9. )*density;
f(:,:,3) = (1./9. )*density;
f(:,:,4) = (1./9. )*density;
f(:,:,5) = (1./9. )*density;
f(:,:,6) = (1./36.)*density;
f(:,:,7) = (1./36.)*density;
f(:,:,8) = (1./36.)*density;
f(:,:,9) = (1./36.)*density;

% define lattice velocity vectors 
ex=zeros(9,1);
ey=zeros(9,1);
ex(0+1)= 0; ey(0+1)= 0;
ex(1+1)= 1; ey(1+1)= 0;
ex(2+1)= 0; ey(2+1)= 1;
ex(3+1)=-1; ey(3+1)= 0;
ex(4+1)= 0; ey(4+1)=-1;
ex(5+1)= 1; ey(5+1)= 1;
ex(6+1)=-1; ey(6+1)= 1;
ex(7+1)=-1; ey(7+1)=-1;
ex(8+1)= 1; ey(8+1)=-1;

%(each point has x component velocity ex
% and y componend ey
u_x = zeros(ny,nx);
u_y = zeros(ny,nx);

% node density
rho = zeros(ny,nx);

for ts=1:tf %Time loop
    
    ts
    
    % Computing macroscopic density, rho, and velocity, u=(ux,uy).
    for j=1:ny 
        
        for i=1:nx
            
            u_x(j,i) = 0.0;
            u_y(j,i) = 0.0;
            rho(j,i) = 0.0;
            
            if ~is_solid_node(j,i)  %fluid nodes
                
                for a=0:8
                    
                    rho(j,i) = rho(j,i) + f(j,i,a+1);
                                       
                    u_x(j,i) = u_x(j,i) + ex(a+1)*f(j,i,a+1);
                    u_y(j,i) = u_y(j,i) + ey(a+1)*f(j,i,a+1);
                    
                end
                
                u_x(j,i) = u_x(j,i)/rho(j,i);
                u_y(j,i) = u_y(j,i)/rho(j,i);
                
            end
                        
        end
    end
    
       
    % Compute the equilibrium distribution function, feq.
    f1=3.;
    f2=9./2.;
    f3=3./2.;
    
    for j=1:ny
        
        for i=1:nx
            
            if ~is_solid_node(j,i)
                
                rt0 = (4./9. )*rho(j,i);
                rt1 = (1./9. )*rho(j,i);
                rt2 = (1./36.)*rho(j,i);
                ueqxij =  u_x(j,i)+tau*g; %add forcing to velocity
                ueqyij =  u_y(j,i);
                uxsq   =  ueqxij * ueqxij;
                uysq   =  ueqyij * ueqyij;
                uxuy5  =  ueqxij +  ueqyij;
                uxuy6  = -ueqxij +  ueqyij;
                uxuy7  = -ueqxij + -ueqyij;
                uxuy8  =  ueqxij + -ueqyij;
                usq    =  uxsq + uysq;
                
                feq(j,i,0+1) = rt0*( 1.                              - f3*usq);
                feq(j,i,1+1) = rt1*( 1. + f1*ueqxij + f2*uxsq        - f3*usq);
                feq(j,i,2+1) = rt1*( 1. + f1*ueqyij + f2*uysq        - f3*usq);
                feq(j,i,3+1) = rt1*( 1. - f1*ueqxij + f2*uxsq         - f3*usq);
                feq(j,i,4+1) = rt1*( 1. - f1*ueqyij + f2*uysq         - f3*usq);
                feq(j,i,5+1) = rt2*( 1. + f1*uxuy5  + f2*uxuy5*uxuy5 - f3*usq);
                feq(j,i,6+1) = rt2*( 1. + f1*uxuy6  + f2*uxuy6*uxuy6 - f3*usq);
                feq(j,i,7+1) = rt2*( 1. + f1*uxuy7  + f2*uxuy7*uxuy7 - f3*usq);
                feq(j,i,8+1) = rt2*( 1. + f1*uxuy8  + f2*uxuy8*uxuy8 - f3*usq);
                
            end
        end
    end
        
    % Collision step.
    for j=1:ny
        
        for i=1:nx
            
            if is_solid_node(j,i) 
                
                % Standard bounceback
                
                temp   = f(j,i,1+1); f(j,i,1+1) = f(j,i,3+1); f(j,i,3+1) = temp;
                temp   = f(j,i,2+1); f(j,i,2+1) = f(j,i,4+1); f(j,i,4+1) = temp;
                temp   = f(j,i,5+1); f(j,i,5+1) = f(j,i,7+1); f(j,i,7+1) = temp;
                temp   = f(j,i,6+1); f(j,i,6+1) = f(j,i,8+1); f(j,i,8+1) = temp;
                
            else  
                % Regular collision away from solid boundary
                
                for a=1:9
                    
                    f(j,i,a) = f(j,i,a)-( f(j,i,a) - feq(j,i,a))/tau; 
                    
                end  
                
            end
        end          
    end
        
    % Streaming step; 
    for j=1:ny
        
        if j>1 
            jn = j-1;
        else
            jn = ny;
        end
        
        if j<ny
            jp = j+1;
        else 
            jp = 1;
        end
        
        for i=1:nx
                  
            if i>1
                in = i-1; 
            else 
                in = nx; 
            end
            if i<nx 
                ip = i+1; 
            else 
                ip = 1; 
            end
            
            
            ftemp(j,i,0+1)  = f(j,i,0+1);
            ftemp(j,ip,1+1) = f(j,i,1+1);
            ftemp(jp,i,2+1) = f(j,i,2+1);
            ftemp(j,in,3+1) = f(j,i,3+1);
            ftemp(jn,i ,4+1) = f(j,i,4+1);
            ftemp(jp,ip,5+1) = f(j,i,5+1);
            ftemp(jp,in,6+1) = f(j,i,6+1);
            ftemp(jn,in,7+1) = f(j,i,7+1);
            ftemp(jn,ip,8+1) = f(j,i,8+1);
            
        end
    end
    
    f=ftemp; 
  
    if rem(ts,statIter)==0
        u = sqrt(u_x.^2+u_y.^2);
        vnew = mean(mean(u));
        error = abs(vold-vnew)/vold;
        vold=vnew;
        
        imagesc(u)
        axis equal; 
        axis tight;
        colorbar;
        title('Speed profile');
        drawnow
        

        
        if error<precision
            disp('Simulation has converged');
            break;
        end
    end
    if ts == tf
        disp('Maximum iteration has reached');
    end
        
end

% plot geometry
figure
image(is_solid_node,'CDataMapping','scaled')
        axis tight
        axis equal
hold on;

% plot velocity profile
[x1,y1] = meshgrid(1:10:nx,1:10:ny);
quiver(x1,y1,u_x(1:10:ny,1:10:nx),u_y(1:10:ny,1:10:nx),'color','w','linewidth',1);
title('Velocity profile');

% plot streamlines
figure
image(is_solid_node,'CDataMapping','scaled')
        axis tight
        axis equal
hold on;
% Start a streamline seed every 10 pixels (feel free to change this
starty = 1:10:ny;
startx = ones(size(starty));
h = streamline(u_x,u_y,startx,starty);
set(h,'color','w','linewidth',1);
title('Streamlines');

toc