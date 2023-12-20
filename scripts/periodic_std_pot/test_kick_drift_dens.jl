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

function KickDrift!(k,x0,p0,nt;dens=Float64[],dt=1,traj=false,axps=nothing,axmf=nothing,mff=nothing,mfms=1.0,ms=0.5,msinit=2.0,colinit="r",col="k",cs=[1.0,1.0,1.0])
    x1 = deepcopy(x0)
    p1 = deepcopy(p0)
    x2 = deepcopy(x0)
    p2 = deepcopy(p0)

    density = length(dens)>0
    N = length(x0)
    if density
        Nx = size(dens)[1]
        Nt = size(dens)[2]
        Dt = k.τ/dt
    end
    i=0
    for i in 1:nt
        #print("Kick $i ...")
        KickDriftStep!(x1,p1,x2,p2,k)
        if density
            for j in 1:N
                #println(j,":",(i-1)*Dt," ",x1[j]," ",i*Dt," ",x2[j])
                drawLine!(dens,(i-1)*Dt,(x1[j]/k.L+0.5)*Nx,i*Dt,(x2[j]/k.L+0.5)*Nx,ds=dsX,mody=false,cscale=cs)
            end
        end
        (x1,x2)=(x2,x1)
        (p1,p2)=(p2,p1)
    end

      return (x1,p1)
end


function initKick(K;ϕ=0.0,τ=1.0,h=1.0,L=10.0)::KickData
    KickData(K,ϕ,L,τ,h)
end

function show(dens,ax,k,Nt;Log=true,fac=20,title=nothing)
   id = copy(dens)
   # mx=maximum(dens)
    if Log
        id=log.(id .+ 1e-2)
    end
    maxd=maximum(id)
    mind=minimum(id)
    if Log
        id=min.((id .- mind)./(maxd-mind),1)
    else
        id=min.(fac*(id .- mind)./(maxd-mind),1)
    end
    heatmap!(ax, id[:,:,3])
    return id
end



function runkickdrift(K;Nt=40,ntr=40000,h=1.0)
    k=initKick(K,L=ceil(Int,2*h*K*Nt/2π),h=h)

    nxdens=4001
    Nproτ=ceil(Int,4000/Nt)

    ntraj=ntr
    ntraj1=ceil(Int,ntr/2.0)
    
    ntdens=ceil(Int,Nt*Nproτ)
    dens=zeros(nxdens,ntdens,3)
    
    x0=Array{Float64}(range(0.0,stop=1.0,length=ntraj))
    p0=zeros(ntraj)
    println("Full flow")
    xt,pt=KickDrift!(k,x0,p0,Nt,dens=dens,dt=k.τ*Nt/ntdens;cs=[0, 0., 1.])

    fig = Figure(size=(800, 600))
    ax1= Axis(fig[1, 1], title = "on top of full flow (log)")
    id = show(dens,ax1,k,Nt)
    return fig, id
end

K=3.7
Nt=30
h=1.0

fig, id = runkickdrift(K,Nt=Nt,h=h)
save("density_2.png",fig)
