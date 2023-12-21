# Kick Drift Model

include("DrawLineColor.jl")
using Statistics
using CairoMakie
using JLD2

mutable struct KickData
    K::Float64 # Kickstrenght
    ϕ::Float64 # Phase
    L::Float64 # System width 
    τ::Float64 # kick period
    h::Float64 #
end


function stdmap(x::Float64,p::Float64,ϕ::Float64,K::Float64,h::Float64)
    pn = p + h*K*sin(2π*x+ϕ)/(2π)
    xn = x + h*pn
    return(xn,pn)
end

function KickDriftStep!(x::Array{Float64},p::Array{Float64},xn::Array{Float64},pn::Array{Float64},k::KickData)
    for i in 1:length(x)
          xn[i],pn[i] =stdmap(x[i],p[i],k.ϕ,k.K,k.h)
    end
end

function KickDrift!(k,x0,p0,nt; dt = 1)
    x1 = deepcopy(x0)
    p1 = deepcopy(p0)
    x2 = deepcopy(x0)
    p2 = deepcopy(p0)
    N = length(x0)
    Dt = k.τ/dt
    xt = zeros(nt+1,N)
    pt = zeros(nt+1,N)
    xt[1,:] = x0
    pt[1,:] = p0
    for i in 1:nt
        KickDriftStep!(x1,p1,x2,p2,k)
            # for j in 1:N
            #     xi = (i-1)*Dt,
            #     yi = (x1[j]/k.L+0.5)*Nx,
            #     xf = i*Dt
            #     yf = (x2[j]/k.L+0.5)*Nx
            # end
        xt[i+1,:]=x2
        pt[i+1,:]=p2
        x1=x2
        p1=p2
    end
  return xt,pt
end


function initKick(K;ϕ=0.0,τ=1.0,h=1.0,L=10.0)::KickData
    KickData(K,ϕ,L,τ,h)
end


function runkickdrift(K;Nt=40,ntr=4000,h=1.0)
    k=initKick(K,L=ceil(Int,2*h*K*Nt/2π),h=h)
    ntraj=ntr
    x0=Array{Float64}(range(0.0,stop=1.0,length=ntraj))
    p0=zeros(ntraj)
    println("Full flow")
    xt,pt=KickDrift!(k,x0,p0,Nt)
    return xt, pt
end


using Interpolations 

function interp_x(itp_x,t) 
    xit = [ x(t) for x in itp_x]
    return xit
end


K=3.7
Nt=30
h=1.0




xt, pt = runkickdrift(K,Nt=Nt,h=h)

# Interpolate each time series between time points 0:Nt
itp_x = [linear_interpolation(0:Nt, x) for x in eachcol(xt)]
itp_p = [linear_interpolation(0:Nt, p) for p in eachcol(pt)]

# Animation begins
time = Observable(0.)
xs_1 = @lift(mod.(interp_x(itp_x,$time),1))
ys_1 = @lift(interp_x(itp_p,$time))
fig = scatter(xs_1, ys_1, color = :blue, linewidth = 4, 
axis = (title = @lift("t = $(round($time, digits = 1))"),))
framerate = 10
timestamps = range(0, Nt, step=0.05)
record(fig, "time_animation.mp4", timestamps;
        framerate = framerate) do t
    time[] = t
end


