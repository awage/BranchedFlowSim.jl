#Draw line routine

@inline function dsLen(dx,dy)
    sqrt(dx^2+dy^2)
end

@inline function dsX(dx,dy)
    abs(dx)
end

@inline function dsY(dx,dy)
    abs(dy)
end

function Add!(A::Array{Float64,3},perm::Bool,i::Int,j::Int,a::Float64,cscale::Vector{Float64})
    for c in 1:3
        (perm) ? A[i,j,c]+=a*cscale[c] : A[j,i,c]+=a*cscale[c];
    end
end

function step_up!(A::Array{Float64,3},x::Float64,y::Float64,dx::Float64,dy::Float64,m::Float64,ds::F,mody::Int64,perm::Bool,cscale::Vector{Float64}) where {F}
    N=size(A,1)
    M=size(A,2)
    if perm
        N,M=M,N
    end
    xf=floor(Int,x)
    yf=floor(Int,y)
    yc=yf+1
    yn=y+dy
    if yc>=yn
        w1,w2=ds(dx,dy),0.0
    else
        dy1=yc-y
        dy2=yn-yc
        dx1=dy1/m
        dx2=dy2/m
        w1,w2=ds(dx1,dy1),ds(dx2,dy2)
    end
    if mody==1
        yf=mod(yf,N)
        yc=mod(yc,N)
    end
    if mody==2
        xf=mod(xf,M)
    end

    if xf<M && xf>=0
        if yf<N && yf>=0
            Add!(A,perm,xf+1,yf+1,w1,cscale)
            #A[yf+1,xf+1]+=w1
        end
        if  yc<N && yc>=0
            Add!(A,perm,xf+1,yc+1,w2,cscale)
            #A[yc+1,xf+1]+=w2
        end
    end
end

function step_down!(A::Array{Float64,3},x::Float64,y::Float64,dx::Float64,dy::Float64,m::Float64,ds::F,mody::Int64,perm::Bool,cscale::Vector{Float64}) where {F}
    N=size(A,1)
    M=size(A,2)
    if perm
        N,M=M,N
    end
    xf=floor(Int,x)
    yf=floor(Int,y)
    yc=yf-1
    yn=y+dy
    if yf<=yn
        w1,w2=ds(dx,dy),0.0
    else
        dy1=y-yf
        dy2=yf-yn
        dx1=dy1/abs(m)
        dx2=dy2/abs(m)
        w1,w2=ds(dx1,dy1),ds(dx2,dy2)
    end
    if mody==1
        yf=mod(yf,N)
        yc=mod(yc,N)
    end
    if mody==2
        xf=mod(xf,M)
    end

    if xf<M && xf>=0
        if yf<N && yf>=0
            #A[yf+1,xf+1]+=w1
            Add!(A,perm,xf+1,yf+1,w1,cscale)
        end
        if  yc<N && yc>=0
            #A[yc+1,xf+1]+=w2
            Add!(A,perm,xf+1,yc+1,w2,cscale)
        end
    end
end

function step!(A::Array{Float64,3},x::Float64,y::Float64,dx::Float64,dy::Float64,m::Float64,ds::F,mody::Int64,perm::Bool,cscale::Vector{Float64}) where {F}
    #println(x," ",y," ",dx," ",dy)
    if m<0
        step_down!(A,x,y,dx,dy,m,ds,mody,perm,cscale)
    else
        step_up!(A,x,y,dx,dy,m,ds,mody,perm,cscale)
    end
end

function drawit!(A::Array{Float64,3},x0::Float64,y0::Float64,x1::Float64,y1::Float64,ds::F,mody::Int64,perm::Bool,cscale::Vector{Float64}) where {F}
    DX=x1-x0
    if DX==0
        return
    end
    DY=(y1-y0)

    m=DY/(1.0*DX)

    x=x0
    y=y0

    if x0<ceil(x0)
        dx=ceil(x)-x0
        dy=m*dx
        step!(A,float(x),y,dx,dy,m,ds,mody,perm,cscale)
        y+=dy
    end
    dx=1.
    dy=m
    xe=floor(Int,x1)
    for x in range(ceil(Int,x0),stop=xe-1)
        step!(A,float(x),y,dx,dy,m,ds,mody,perm,cscale)
        y+=dy
    end

    if (xe<x1)
        dx=x1-xe
        dy=m*dx
        step!(A,float(xe),y,dx,dy,m,ds,mody,perm,cscale)
    end
end

function drawLine!(A::Array{Float64,3},x0::Float64,y0::Float64,x1::Float64,y1::Float64;ds::F=dsLen,mody=false,cscale=[1.0,1.0,1.0]) where {F}
    perm=false
    dx=x1-x0
    dy=y1-y0
    mod=0
    if abs(dx)>=abs(dy)
       a0,a1,b0,b1=x0,x1,y0,y1
       if mody
           mod=1
       end
   else
       a0,a1,b0,b1=y0,y1,x0,x1
       perm=true
       if ds==dsX
           ds=dsY
       end
       if mody
           mod=2
       end
   end
   if (a0>a1)
       a0,a1,b0,b1=a1,a0,b1,b0
   end
   drawit!(A,a0,b0,a1,b1,ds,mod,perm,cscale)
end
