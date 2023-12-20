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


function runkickdrift(K;Nt=40,ntr=400,h=1.0)
    k=initKick(K,L=ceil(Int,2*h*K*Nt/2π),h=h)
    ntraj=ntr
    x0=Array{Float64}(range(0.0,stop=1.0,length=ntraj))
    p0=zeros(ntraj)
    println("Full flow")
    yt,pt=KickDrift!(k,x0,p0,Nt)
    return yt, pt
end

K=3.7
Nt=30
h=1.0

yt, pt = runkickdrift(K,Nt=Nt,h=h)

time = Observable(1)


xs_1 = @lift(yt[$time,:])
ys_1 = @lift(pt[$time,:])

fig = scatter(xs_1, ys_1, color = :blue, linewidth = 4)

framerate = 5
timestamps = range(1, Nt, step=1)

record(fig, "time_animation.mp4", timestamps;
        framerate = framerate) do t
    time[] = t
end


