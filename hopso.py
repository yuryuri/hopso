# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:30:57 2025

@author: mirac
"""

def hopso(cost_fn, hp, num_particles, runs, dimension,max_cut,e_min, vectors, velocities, vel_mag, gbest,amps,pos, max_iterations=1000):
    import numpy as np
    from tqdm import tqdm
    # Calculate constants
    omega = 1
    lamb = hp[3]
    tm = hp[2]
    
    
    for _ in tqdm(range(runs)):
        # INITIALIZE PARTICLES
        
        particles_position = np.array(np.random.uniform(0, np.pi, size=(num_particles, dimension))) #CHANGE BOUNDS ACCORDINGLY
        particles_velocity = np.array(np.random.uniform(-1, 1, size=(num_particles, dimension))) #CHANGE BOUNDS ACCORDINGLY
        
        # INITIALIZE PERSONAL BEST POSITION
        personal_best_positions = particles_position.copy()
        
        # INITIALIZE VELOCITY MAGNITUDES
        velocity_magnitudes = np.zeros((max_iterations, num_particles)) #For analysing velocities
        
        # EVALUATE THE FUNCTION AT INITIAL POSITIONS
        personal_best_values = np.array([cost_fn(p) for p in personal_best_positions])
        global_best_index = np.argmin(personal_best_values)
        global_best_energy = cost_fn(personal_best_positions[global_best_index])
        global_best_position = personal_best_positions[global_best_index].copy()
        gb_position = np.zeros((num_particles, dimension))
        gb_position[:] = global_best_position 
        gb = []
        
        # Initialize iteration counter
        iteration = 0
        pbcount = []
        gbcount = []
        # CALCULATE INITIAL ATTRACTORS
        attractors = (hp[0] * personal_best_positions + hp[1] * gb_position) / (hp[0] + hp[1])
      
        #Calculate initial amplitude and intial angle
        A = np.sqrt((particles_position-attractors)**2+(1/(omega))**2*(particles_velocity+(lamb)*(particles_position-attractors))**2)
        amp_dis = np.abs(personal_best_positions - gb_position)/2
        amp_dis = (amp_dis)*max_cut
        A = np.maximum(A, amp_dis)
        theta = np.arccos((particles_position-attractors)/A)
        
        # Update particle velocities and positions

        A_list = []
        x_list = []
        v_list = []
        A1 = [[] for _ in range(num_particles)]
        amp_dist = [[] for _ in range(num_particles)]
        tim = []

        t = np.zeros((num_particles,dimension))
        
        #Main loop
        while iteration < max_iterations: 
            t += np.random.rand(num_particles,dimension)*tm

            A = A* np.exp(-lamb * t) 
            a_dist = (np.abs(personal_best_positions - gb_position))/2
            a_dist = a_dist*max_cut
            A = np.maximum(A,a_dist)

            particles_position = (A * np.cos(omega * t  + theta)) + attractors
            particles_position = np.clip(particles_position, 0, np.pi)
            x_list.append(particles_position )

            particles_velocity = A * (-omega * np.sin(omega *t  + theta) - lamb * np.cos(omega * t + theta))
            v_list.append(particles_velocity)
            
          
            
            # Update personal best positions
            for i in range(num_particles):
                current_value = cost_fn(particles_position[i])
                if current_value < personal_best_values[i]:
                    personal_best_positions[i] = particles_position[i]
                    personal_best_values[i] = current_value
                    t[i] = 0  # Reset t for the updated particle
                    #Recalculate attractors, amplitude and angle for that specific particle
                    attractors[i] = (hp[0] * personal_best_positions[i] + hp[1] * gb_position[i]) / (hp[0] + hp[1])
                    amp_dist[i] = np.abs(personal_best_positions[i] - gb_position[i])/2
                    amp_dist[i] = (amp_dist[i])*max_cut
                    A1[i] = np.sqrt((particles_position[i] - attractors[i]) ** 2 + (1/(omega))** 2 * (particles_velocity[i] + lamb* (particles_position[i] - attractors[i])) ** 2)
                    A[i] = np.maximum(A[i],A1[i],amp_dist[i])
                    theta[i] = np.arccos((particles_position[i] - attractors[i]) / A[i])
                    pbcount.append((iteration,i))
            
            if np.min(personal_best_values) < global_best_energy:
                gbcount.append(iteration)
                global_best_index = np.argmin(personal_best_values)
                global_best_position = personal_best_positions[global_best_index].copy()
                global_best_energy = personal_best_values[global_best_index]
                gb_position[:] = global_best_position 
                t[:] = 0 # Reset time for all particles
                # Recalculate attractors, amplitude and angle for all particles
                attractors = (hp[0] * personal_best_positions + hp[1] * gb_position) / (hp[0] + hp[1])
                amp_dist_all = np.abs(personal_best_positions - gb_position)/2
                amp_dist_all = (amp_dist_all)*max_cut
                A1 = np.sqrt((particles_position - attractors) ** 2 + (1/(omega))** 2 * (particles_velocity + lamb * (particles_position - attractors)) ** 2)
                A = np.maximum(A,A1,amp_dist_all)
                theta = np.arccos((particles_position - attractors) / A)
            
            velocity_magnitudes[iteration, :] = np.linalg.norm(particles_velocity, axis=1) # For analysing velocities
            gb.append(cost_fn(global_best_position))
            A_list.append(A)

            # Increment iteration counter
            tim.append(t.copy())
            
            # # Increment iteration counter
            iteration += 1
        # Append results    
        pos.append(x_list)
        velocities.append(v_list)
        amps.append(A_list)
        gbest.append(gb)
        e_min.append(np.float64(cost_fn(global_best_position)))
        vectors.append(global_best_position)
        vel_mag.append(velocity_magnitudes)
