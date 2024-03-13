using ChaosTools
using LinearAlgebra:norm

function transverse_stretching(ds::DynamicalSystem, T; u0 = initial_state(ds), e = 1e-6, dir = 1)
    Δt = 1
    Q0 = [e, 0]
    states = [u0, u0 .+ Q0]
    pds = ParallelDynamicalSystem(ds, states)
    d0 = e
    d0_upper = d0*1e3
    λlocal = 0
    while current_time(pds) <  T
        d = current_state(pds, 2) .- current_state(pds, 1)
        d = abs(d[dir]) # only the variable y!
        while d ≤ d0_upper 
            step!(pds, Δt)
            d = current_state(pds, 2) .- current_state(pds, 1)
            d = abs(d[dir]) # only the variable y!
            current_time(pds) ≥  T && break
        end
        a = d/d0
        λlocal += log(a)
        
        u1 = current_state(pds, 1)
        u2 = current_state(pds, 2)
        @. u2 = u1 + (u2 - u1)/a
        set_state!(pds, u2, 2)
    end
    d = current_state(pds, 2) .- current_state(pds, 1)
    d = abs(d[dir]) # only the variable y!
    a = d/d0
    λlocal += log(a)
    return λlocal
end

