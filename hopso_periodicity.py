def hopso(
    cost_fn,
    hp,
    num_particles,
    runs,
    dimension,
    max_cut,
    e_min,
    vectors,
    velocities,
    vel_mag,
    gbest,
    max_iterations=500
):
    """
    Demonstration: each particle updates its attractor/amplitude/theta
    IMMEDIATELY after it finds a better personal best.
    Then, at the end of each iteration, we check global best and do
    a SWARM-WIDE attractor/amplitude/theta update if there's a new global best.
    """

    import numpy as np
    from tqdm import tqdm

    # Unpack hyperparameters
    w1   = hp[0]
    w2   = hp[1]
    tm   = hp[2]
    lamb = hp[3]

    omega = 1.0

    for run_idx in range(runs):
        r = np.random.uniform(0, 2*np.pi)

        # Initialize positions and velocities
        positions = np.random.uniform(r, r + 2*np.pi, size=(num_particles, dimension))
        particle_vels = np.random.uniform(-np.pi/2, np.pi/2, size=(num_particles, dimension))

        # Personal best
        personal_best_positions = positions.copy()
        personal_best_values = np.array([cost_fn(p) for p in positions])

        # Global best
        gbest_idx = np.argmin(personal_best_values)
        global_best_value = personal_best_values[gbest_idx]
        global_best_position = personal_best_positions[gbest_idx].copy()

        # Histories
        gb_run_history = []

        # Particle time, amplitude, attractors, theta
        t        = np.zeros((num_particles, dimension))
        dead     = np.zeros(num_particles, dtype=bool)

        # Compute initial attractors
        delta = np.abs(personal_best_positions - global_best_position)
        mask  = delta > np.pi
        attractors = np.where(
            mask,
            np.mod(
                ((w1 * personal_best_positions + w2 * global_best_position) / (w1 + w2))
                + np.pi - r,
                2*np.pi
            ) + r,
            (w1 * personal_best_positions + w2 * global_best_position) / (w1 + w2)
        )

        # Initial amplitude
        A = np.sqrt((positions - attractors)**2
                    + (1/omega)**2 * (particle_vels + lamb*(positions - attractors))**2)
        amp_dis = (delta % (2*np.pi))
        amp_dis = (np.minimum(2*np.pi - amp_dis, amp_dis)/2)*max_cut
        A = np.maximum(A, amp_dis)

        # Initial theta
        cos_theta = (positions - attractors)/A
        theta = np.zeros_like(cos_theta)
        for i in range(num_particles):
            # If out of bounds => kill
            if (np.any(cos_theta[i] < -1) or np.any(cos_theta[i] > 1) or np.isnan(cos_theta[i]).any()):
                personal_best_values[i] = np.inf
                dead[i] = True
                print(f"[Run {run_idx}] Particle {i} killed at init.")
            else:
                theta[i] = np.arccos(cos_theta[i])

        # For optional velocity magnitudes
        velocity_magnitudes = np.zeros((max_iterations, num_particles))

        for iteration in tqdm(range(max_iterations), desc=f"Run {run_idx}", leave=True):
            # -------------
            # (A) Evolve each particle
            # -------------
            for i in range(num_particles):
                if dead[i]:
                    continue

                # Evolve time & amplitude
                delta_t = np.random.rand(dimension)*tm
                t[i] += delta_t
                A[i] *= np.exp(-lamb * delta_t)

                # Enforce min distance
                # Based on that particle's pbest vs global best
                d_i = np.abs(personal_best_positions[i] - global_best_position) % (2*np.pi)
                a_dist = (np.minimum(2*np.pi - d_i, d_i)/2)*max_cut
                A[i] = np.maximum(A[i], a_dist)

                # Update position & velocity
                positions[i] = A[i]*np.cos(omega*t[i] + theta[i]) + attractors[i]
                particle_vels[i] = A[i]*(-omega*np.sin(omega*t[i] + theta[i])
                                         - lamb*np.cos(omega*t[i] + theta[i]))

                # Evaluate cost
                current_value = cost_fn(positions[i])

                # -------------
                # (A.1) If there's a personal best update,
                # we IMMEDIATELY recalc attractors/amplitude/theta
                # for that single particle
                # -------------
                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = np.mod(positions[i] - r, 2*np.pi) + r
                    t[i] = 0  # reset time

                    # Now update attractor dimensionwise for this particle
                    delta_p = np.abs(personal_best_positions[i] - global_best_position)
                    mask_p  = delta_p > np.pi
                    new_attractor = np.where(
                        mask_p,
                        np.mod(
                            ((w1 * personal_best_positions[i] + w2 * global_best_position) / (w1 + w2)) + np.pi - r,
                            2*np.pi
                        ) + r,
                        (w1 * personal_best_positions[i] + w2 * global_best_position) / (w1 + w2)
                    )
                    attractors[i] = new_attractor

                    # Recompute amplitude for that particle
                    A1_i = np.sqrt(
                        (positions[i] - attractors[i])**2
                        + (1/omega)**2 * (particle_vels[i] + lamb*(positions[i] - attractors[i]))**2
                    )
                    # also enforce min distance again
                    a_dist_i = (delta_p % (2*np.pi))
                    a_dist_i = (np.minimum(2*np.pi - a_dist_i, a_dist_i)/2)*max_cut
                    A[i] = np.maximum(np.maximum(A[i], A1_i), a_dist_i)

                    # Recompute cos_theta for that particle
                    cos_th_i = (positions[i] - attractors[i])/A[i]
                    # Kill if invalid
                    if (np.any(cos_th_i < -1) or np.any(cos_th_i > 1) or np.isnan(cos_th_i).any()):
                        personal_best_values[i] = np.inf
                        dead[i] = True
                        print(f"[Run {run_idx}] Particle {i} killed at iteration {iteration} after pbest update.")
                    else:
                        theta[i] = np.arccos(cos_th_i)

            # -------------
            # (B) After all personal best updates, check global best
            # -------------
            current_best_idx = np.argmin(personal_best_values)
            current_best_val = personal_best_values[current_best_idx]
            if current_best_val < global_best_value:
                global_best_value    = current_best_val
                global_best_position = personal_best_positions[current_best_idx].copy()
                # Reset all times
                t[:] = 0

                # -------------
                # (B.1) Now do a swarm-wide attractor & amplitude update
                # because the global best changed
                # -------------
                delta_all = np.abs(personal_best_positions - global_best_position)
                mask_all  = (delta_all > np.pi)
                attractors = np.where(
                    mask_all,
                    np.mod(
                        ((w1*personal_best_positions + w2*global_best_position)/(w1+w2))
                        + np.pi - r, 2*np.pi
                    ) + r,
                    (w1*personal_best_positions + w2*global_best_position)/(w1+w2)
                )
                # Recompute amplitude & kill invalid
                A_all = np.sqrt(
                    (positions - attractors)**2
                    + (1/omega)**2 * (particle_vels + lamb*(positions - attractors))**2
                )
                a_dist_all = (delta_all % (2*np.pi))
                a_dist_all = (np.minimum(2*np.pi - a_dist_all, a_dist_all)/2)*max_cut
                A = np.maximum(np.maximum(A, A_all), a_dist_all)

                # Update cos_theta for all
                cos_theta = (positions - attractors)/A
                for i in range(num_particles):
                    if dead[i]:
                        continue
                    if (np.any(cos_theta[i] < -1) or np.any(cos_theta[i] > 1) or np.isnan(cos_theta[i]).any()):
                        personal_best_values[i] = np.inf
                        dead[i] = True
                        print(f"[Run {run_idx}] Particle {i} killed after global best updated.")
                    else:
                        theta[i] = np.arccos(cos_theta[i])

            # (C) Record velocity magnitudes if you like
            velocity_magnitudes[iteration] = np.linalg.norm(particle_vels, axis=1)

            # (D) Save global best for iteration
            gb_run_history.append(global_best_value)

        # End of run: store run-level results
        e_min.append(global_best_value)
        vectors.append(global_best_position)
        velocities.append(particle_vels)
        vel_mag.append(velocity_magnitudes)
        gbest.append(gb_run_history) 