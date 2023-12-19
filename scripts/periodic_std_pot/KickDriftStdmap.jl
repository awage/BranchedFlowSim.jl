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
    if traj
        xt = zeros(nt+1,N)
        pt = zeros(nt+1,N)
        xt[1,:] = x0
        pt[1,:] = p0
    end

    if axmf != nothing 
        if mff == nothing
            mffac = 1.0
        else 
            mffac = 1.0*mff
        end
        axmf.plot(x0, p0,".",markersize=ms,color="k")
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
            if axps!=nothing && col !=nothing
                axps.plot(mod.(x2,1), mod.(p2 .+ 0.5/h,1/h) .- 0.5/h,".",markersize=ms,color=col,markeredgecolor="none",markeredgewidth=0)
            end

            if axmf!=nothing 
                axmf.plot(x2, p2*mffac .+ i,".",markersize=mfms,markeredgecolor="none",markeredgewidth=0)
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
        axps.plot(mod.(x0,1),mod.(p0 .+ 0.5/h,1/h) .- 0.5/h,".",markersize=msinit,color=colinit,markeredgecolor="none",markeredgewidth=0)
    end
    axps.set_xlim([0,1])
    axps.set_ylim([-0.5/h,0.5/h])
    if traj
      return (xt,pt)
  else
      return (x1,p1)
  end
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
    extent=[-k.L/2,k.L/2,0,Nt]
    ax.imshow(permutedims(id,(2,1,3)),extent=extent,aspect="auto",origin="lower")
    if title!== nothing
        ax.set_title(title)
    end
end



function runkickdrift(K;Nt=40,save=false,dpi=400,ntr=10000,h=1.0)
    # gs=nothing
    # el(i,j) = get(gs, (i,j))
    # sl(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
    
    #Nt=40
    k=initKick(K,L=ceil(Int,2*h*K*Nt/2π),h=h)
    #k=initKick(K,L=ceil(Int,10),h=h)

    nxdens=4001
    Nproτ=ceil(Int,4000/Nt)

    ntraj=ntr
    ntraj1=ceil(Int,ntr/2.0)
    
    ntdens=ceil(Int,Nt*Nproτ)
    dens=zeros(nxdens,ntdens,3)

    nps=20
    xps=[x for x= range(0,1,length=nps),y=range(0,1,length=nps) .- 0.5]
    pps=[ y/h for x=range(0,1,length=nps),y=range(0,1,length=nps) .- 0.5]
    
    x0=Array{Float64}(range(0.0,stop=1.0,length=ntraj))
    p0=zeros(ntraj)
    x1=Array{Float64}(range(0.02,stop=0.05,length=ntraj1))
    p1=zeros(ntraj1)

    close("all")
    #fig=figure(figsize=(24,6))
    fig,(ax1,ax2,ax3,axps)=subplots(1,4,figsize=(24,6))
    figmf,(axmf1,axmf2)=subplots(1,2,figsize=(12,6))
    # gs = PyPlot.matplotlib.gridspec.GridSpec(2, 4, figure=fig)
    # ax1=fig.add_subplot(el(0,0))
    # ax2=fig.add_subplot(el(0,1))
    # ax3=fig.add_subplot(el(1,0))
    # axps=fig.add_subplot(el(1,1))
    # axmf=fig.add_subplot(el(sl(0,2),sl(2,5)))
    


    println("Illustrate Phasespace")
    KickDrift!(k,xps,pps,500,dt=k.τ*Nt/ntdens,axps=axps,col="k",colinit=nothing)
    println("Full flow")
    xt,pt=KickDrift!(k,x0,p0,Nt,dens=dens,dt=k.τ*Nt/ntdens,axps=axps,col=nothing,colinit="g",cs=[0,1,1.0],axmf=axmf1,mff=1.0/3 )
    println("Partial flow")

    dens1=zeros(size(dens))
    xt,pt=KickDrift!(k,x1,p1,Nt,dens=dens1,dt=k.τ*Nt/ntdens,axps=axps,col=nothing,colinit="r",cs=[1.0,0,0],axmf=axmf2,mff=1.0/3)
    show(dens .+ dens1,ax1,k,Nt,title="on top of full flow (log)")
    show(dens1,ax2,k,Nt,title="logarithmic")
    show(dens1,ax3,k,Nt,Log=false,fac=40,title="linear")

    date=Dates.format(Dates.now(),"yyyy-mm-dd-HH-MM-SS")

    #@save "example-$date.jld2" dens ϵ ℓ τ ntraj
    
    fig.tight_layout()
    figmf.tight_layout()

    if save
        name="Stdmap-$(date)-K$(K).png"
        namemf="Stdmap-$(date)-K$(K)-mf.png"
        fig.savefig(name,dpi=dpi)
        figmf.savefig(namemf,dpi=dpi)
        
    end

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

K=3.7
Nt=30
#h=1/3
h=1.0
#save=true
save=false

@time runkickdrift(K,Nt=Nt,save=save,h=h)
