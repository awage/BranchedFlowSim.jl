# Kick Drift Model
# include("DrawLineColor.jl")
using CairoMakie 
using Statistics
using JLD2, FileIO
using Dates


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

function KickDrift!(k,x0,p0,nt;axmf=nothing,mff=nothing,mfms=1.0, ms = 0.5)
    x1 = deepcopy(x0)
    p1 = deepcopy(p0)
    x2 = deepcopy(x0)
    p2 = deepcopy(p0)

    if axmf != nothing 
        if mff == nothing
            mffac = 1.0
        else 
            mffac = 1.0*mff
        end
        plot!(axmf,x0, p0; markersize=ms)
    end
    
    for i in 1:nt
            KickDriftStep!(x1,p1,x2,p2,k)
            if axmf!=nothing 
                plot!(axmf, x2, p2*mffac .+ i; markersize=mfms,markeredgecolor="none",markeredgewidth=0)
            end
            x1 = x2; p1 = p2
    end

  return (x1,p1)
end

function initKick(K;ϕ=0.0,τ=1.0,h=1.0,L=10.0)::KickData
    KickData(K,ϕ,L,τ,h)
end

function runkickdrift(K;Nt=40,save=false,dpi=400,ntr=1000,h=1.0)
    k=initKick(K,L=ceil(Int,2*h*K*Nt/2π),h=h)
    x0=Array{Float64}(range(0.0,stop=1.0,length=ntr))
    p0=zeros(ntr)
    fig = Figure(resolution=(800, 600))
    axmf1= Axis(fig[1, 1])
    xt,pt=KickDrift!(k,x0,p0,Nt;axmf=axmf1,mff=1.0/10 )
    return fig
end

K=3.7
Nt=10
h=1.0

fig = runkickdrift(K,Nt=Nt,h=h)
