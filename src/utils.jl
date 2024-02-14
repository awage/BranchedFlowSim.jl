using ChaosTools
using LinearAlgebra:norm

function transverse_growth_rates(ds::DynamicalSystem, points;
        S = 100, Δt = 5, e = 1e-6,
        perturbation = (ds, u, j) -> _random_Q0(ds, u, j, e),
        dir = 1 # transverse dir
    )

    Q0 = [randn()*e, 0]
    states = [points[1], points[1] .+ Q0]
    pds = ParallelDynamicalSystem(ds, states)
    # Function barrier
    return transverse_growth_rates(pds, dir, points, ds, S, Δt, e)
end

function transverse_growth_rates(pds, dir, points, ds, S, Δt, e)
    λlocal = zeros(length(points), S)
    pds_array = [deepcopy(pds) for i in 1:S]
    for (i, u) in enumerate(points)
        Threads.@threads for j in 1:S
            Q0 = [randn()*e, 0]
            states1 = u
            states2 = states1 .+ Q0
            g0 = norm(Q0)
            reinit!(pds_array[j], [states1, states2])
            step!(pds_array[j], Δt, true)
            g = current_state(pds_array[j], 2) .- current_state(pds_array[j], 1)
            g = abs(g[dir]) # only the variable y!
            tspan = (current_time(pds_array[j]) - initial_time(pds_array[j]))
            λlocal[i, j] = log(g/g0)/tspan
        end
    end
    return λlocal
end

