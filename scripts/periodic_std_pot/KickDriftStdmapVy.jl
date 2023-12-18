# Kick Drift Model

include("DrawLineColor.jl")
#using FFTW

const eps=1e-8

using PyCall
#pygui(:tk)
using PyPlot
#pygui(true)
using Statistics
using JLD2, FileIO
using Dates


 mutable struct KickData
    K::Float64 # Kickstrenght
    ϕ::Float64 # Phase
    L::Float64 # System width 
    τ::Float64 # kick period
end


function stdmap(x::Float64,p::Float64,ϕ::Float64,K::Float64)
    pn = p + K*sin(2π*x+ϕ)/(2π)
    xn = x + pn
    return(xn,pn)
end


function KickDriftStep!(x::Array{Float64},p::Array{Float64},xn::Array{Float64},pn::Array{Float64},k::KickData)
    for i in 1:length(x)
          xn[i],pn[i] =stdmap(x[i],p[i],k.ϕ,k.K)
    end
end

function KickDrift!(k,x0,p0,nt;vy=nothing,dens=Float64[],dt=1,traj=false,axps=nothing,ms=0.5,msinit=4.0,colinit="r",col="k",cs=[1.0,1.0,1.0])
    x1=deepcopy(x0)
    p1=deepcopy(p0)
    x2=deepcopy(x0)
    p2=deepcopy(p0)

    density=length(dens)>0
    N=length(x0)
    if density
        Nx=size(dens)[1]
        Nt=size(dens)[2]
        Dt=k.τ/dt
    end
    if traj
        xt=zeros(nt+1,N)
        pt=zeros(nt+1,N)
        xt[1,:]=x0
        pt[1,:]=p0
    end

    
    i=0
    for i in 1:nt
            #print("Kick $i ...")
            KickDriftStep!(x1,p1,x2,p2,k)


            if density
                v=1.0
                for j in 1:N
                    if vy!=nothing
                        v=vy[j]
                    end
                    #println(j,":",(i-1)*Dt," ",x1[j]," ",i*Dt," ",x2[j])
                    drawLine!(dens,(i-1)*v*Dt,(x1[j]/k.L+0.5)*Nx,i*v*Dt,(x2[j]/k.L+0.5)*Nx,ds=dsX,mody=false,cscale=cs)
                end
            end
            if axps!=nothing && col !=nothing
                axps.plot(mod.(x2,1), mod.(p2 .+ 0.5,1) .- 0.5,".",markersize=ms,color=col)
            end
            if traj
                xt[i+1,:]=x2
                pt[i+1,:]=p2
            end
            (x1,x2)=(x2,x1)
            (p1,p2)=(p2,p1)
            #println("done")
    end

    if axps!=nothing && colinit !=nothing
        axps.plot(mod.(x0,1),mod.(p0 .+ 0.5,1) .- 0.5,".",markersize=msinit,color=colinit)
    end
    axps.set_xlim([0,1])
    axps.set_ylim([-0.5,0.5])
    if traj
      return (xt,pt)
  else
      return (x1,p1)
  end
end


function initKick(K;ϕ=0.0,τ=1.0,L=10.0)::KickData
    KickData(K,ϕ,L,τ)
end


function runkickdrift(K)
    k=initKick(K,L=50)


    Nt=21
    Nproτ=100
    ntraj=10000
    nxdens=4001
    ntdens=ceil(Int,Nt*Nproτ)
    dens=zeros(nxdens,ntdens,3)

    nps=20
    xps=[x for x= range(0,1,length=nps),y=range(0,1,length=nps) .- 0.5]
    pps=[ y for x=range(0,1,length=nps),y=range(0,1,length=nps) .- 0.5]
    
    x0=Array{Float64}(range(0.0,stop=1.0,length=ntraj))

    v=sqrt.(2*K*(2 .+ cos.(2π*x0))/5)/(2π) #.* sign.(-sin.(2π*x0))
    p0=v
   

   
    x1=Array{Float64}(range(0.02,stop=0.05,length=ntraj))
    v=sqrt.(2*K*(2 .+ cos.(2π*x1))/5)/(2π) #.* sign.(-sin.(2π*x1))
    p1=v
 
    
    close("all")
    fig,(ax1,ax2)=subplots(1,2,figsize=(12,6))

    println("Illustrate Phasespace")
    KickDrift!(k,xps,pps,200,dt=k.τ*Nt/ntdens,axps=ax2,col="k",colinit=nothing,ms=0.1)
    println("Full flow")
    xt,pt=KickDrift!(k,x0,p0,Nt,dens=dens,dt=k.τ*Nt/ntdens,axps=ax2,col=nothing,colinit="g",cs=[0,1,1.0])
    println("Partial flow")
    xt,pt=KickDrift!(k,x1,p1,Nt,dens=dens,dt=k.τ*Nt/ntdens,axps=ax2,col=nothing,colinit="r",cs=[1.0,0,0])



    date=Dates.format(Dates.now(),"yyyy-mm-dd-HH-MM-SS")

    #@save "example-$date.jld2" dens ϵ ℓ τ ntraj
   
    id=dens./mean(dens)
    mx=maximum(dens)
    
    id=log.(id .+ 1e-2)
    maxd=maximum(id)
    mind=minimum(id)
    id=min.((id .- mind)./(maxd-mind),1)
    extent=[-k.L/2,k.L/2,0,Nt]
    ax1.imshow(permutedims(id,(2,1,3)),extent=extent,aspect="auto",origin="lower")
    title(date)
    tight_layout()

    # pm=maximum(pt)
    # m=20
    # figure()
    # for i in 1:m:size(xt)[1]
    #     plot(m.*pt[i,:]./(2.5*pm).+i,xt[i,:],".",color="k")
    # end

    # figure()
    # t=1.0*collect(0:Nt)
    #
    # m=100
    # for i in 1:m:size(xt)[2]
    #     plot(t,xt[:,i],marker=".")
    # end

    # figure()
    # plot(dens[:,floor(Int,ntdens*0.75),1])
    # xlabel("y")
    # ylabel(L"$\rho(y)$")
    # show()
end
K=3.0
@time runkickdrift(K)
