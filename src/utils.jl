using LinearAlgebra:norm

function transverse_growth_rates(ds::DynamicalSystem, points;
        S = 100, Δt = 5, e = 1e-6,
        perturbation = (ds, u, j) -> _random_Q0(ds, u, j, e),
    )

    Q0 = [randn()*e, 0]
    states = [points[1], points[1] .+ Q0]
    pds = ParallelDynamicalSystem(ds, states)
    # Function barrier
    return transverse_growth_rates(pds, states, points, ds, S, Δt, e)
end

function transverse_growth_rates(pds, states, points, ds, S, Δt, e)
    λlocal = zeros(length(points), S)
    for (i, u) in enumerate(points)
        for j in 1:S
            Q0 = [randn()*e, 0]
            states[1] = u
            states[2] = states[1] .+ Q0
            g0 = norm(Q0)
            reinit!(pds, states)
            step!(pds, Δt, true)
            g = current_state(pds, 2) .- current_state(pds, 1)
            g = abs(g[1])
            tspan = (current_time(pds) - initial_time(pds))
            λlocal[i, j] = log(g/g0)/tspan
        end
    end
    return λlocal
end

