using Rotations
using StaticArrays
using LinearAlgebra
using Statistics
using Distributions
using Plots

# original mean in Rotations.jl does not work anymore bc eigfact() fcn in LinAlg changed to eigen()
function Statistics.mean(qvec::AbstractVector{UnitQuaternion{T}}, method::Integer = 0) where T
    #if (method == 0)
    M = zeros(4, 4)
    for i = 1:length(qvec)
        q = qvec[i]
        Qi = @SVector [q.w, q.x, q.y, q.z]  # convert types to ensure we don't get quaternion multiplication
        M .+= Qi * (Qi')
    end
    evec = LinearAlgebra.eigen(Symmetric(M), 4:4)
    Qbar = UnitQuaternion(evec.vectors[1], evec.vectors[2], evec.vectors[3], evec.vectors[4]) # This will renormalize the quaternion...
    #else
    #    error("I haven't coded this")
    #end

    return Qbar
end

# predicts the evolution of the state x based on the input u
# @parameter x state estimate
# @parameter P covariance of state estimate
# @parameter u input is rotational velocity of body wrt. earth in body frame Ï_BE^B
function predict(x::UnitQuaternion, P::SMatrix{3,3}, u::SVector{3}, Q::SMatrix{3,3}, Tâ )
    n = size(P, 1)
    # see E. Kraft "A Quaternion-based Unscented Kalman Filter for Orientation Tracking" for computation of the sigma points 
    ğ¦ = [ sqrt(n*(P + Tâ^2*Q)) (-sqrt(n*(P + Tâ^2*Q)))]
    ğ§ = [ x â Rotations.RotationError(SVector(ğ¦[:,i]), CayleyMap()) for i = 1:2n] # implementation of errormap from Rotations.jl
    
    # integration by direct multiplication method described in section 3.2 of F. Zhao, B.G.M. van Wachem "A novel Quaternion integration approach for describing behaviour of non-spherical particles"
    dq = UnitQuaternion(RotationVec(Tâ*u[1], Tâ*u[2], Tâ*u[3]))
    ğ¨ = [ dq * ğ§[i] for i = 1:2n]
    x = Statistics.mean(ğ¨)
    
    # predicted covariance
    P = 1/(2n)*sum((ğ¨[i] â x)*(ğ¨[i] â x)' for i=1:2n)
    P = SMatrix{3,3}(1/2*(P+P'))
    return x, P
end

# generates an augmented rotational measurement from heading (gravity + north direction)
# see A. C. B. Chiella, B. O.S. Teixeira, G. A. S. Pereira "Quaternion-Based Robust Attitude Estimation Using an Adaptive Unscented Kalman Filter" for details of the measurement model
# @parameter z vector of [g^B; n^B] the gravity vector and north direction in the body fixed frame (normalized)
# @parameter R measurement covariance matrix
function augrotmeas(z::SVector{6}, R::SMatrix{6,6})
    n = size(R,1)
    ğ¦ = [ sqrt(n*(R)) (-sqrt(n*(R)))]
    ğ© = [ z + ğ¦[:,i] for i=1:2n ]
    ğ¨ = [one(UnitQuaternion) for i=1:2n]
    for i = 1:2n
        qâ = one(UnitQuaternion)
        a = ğ©[i][1:3] 
        if a[3] >=0
            qâ = UnitQuaternion(sqrt((a[3]+1)/2), -a[2]/sqrt(2(a[3]+1)), a[1]/sqrt(2(a[3]+1)), 0)
        else 
            qâ = UnitQuaternion(-a[2]/sqrt(2(1-a[3])), sqrt((1-a[3])/2), 0, a[1]/sqrt(2(1-a[3])))
        end
        qâ = one(UnitQuaternion)
        l = qâ^-1*ğ©[i][4:6]
        Î = l[1]^2+l[2]^2
        if l[1] >=0
            qâ = UnitQuaternion(sqrt(Î+l[1]*sqrt(Î))/sqrt(2Î), 0, 0, l[2]/sqrt(2*(Î+l[1]*sqrt(Î))))
        else
            qâ = UnitQuaternion(l[2]/sqrt(2*(Î-l[1]*sqrt(Î))), 0, 0, sqrt(Î-l[1]*sqrt(Î))/sqrt(2Î))
        end
        ğ¨[i] = (qâ*qâ)^-1
    end
    ğ = Statistics.mean(ğ¨)
    ğ¡ = 1/(2n)*sum((ğ¨[i] â ğ)*(ğ¨[i] â ğ)' for i=1:2n)
    return ğ, ğ¡
end

# updates the state x based on measurement z
# @parameter x state estimate
# @parameter P covariance of state estimate
function update(x::UnitQuaternion, P::SMatrix{3,3}, z::SVector{6}, R::SMatrix{6,6})
    n = size(P, 1)
    # see E. Kraft "A Quaternion-based Unscented Kalman Filter for Orientation Tracking" for computation of the sigma points 
    ğ¦ = [ sqrt(n*P) (-sqrt(n*P))]
    ğ§ = [ x â Rotations.RotationError(SVector(ğ¦[:,i]), CayleyMap()) for i = 1:2n] # implementation of errormap from Rotations.jl
    ğ§, ğ¡ = augrotmeas(z, R)

    ğ© = ğ§ # apply h in case the augmented measurement created above is different from the state...
    ğ = Statistics.mean(ğ©) # not required since h for identity, just for clarification
    
    # innovation covariance
    Pvv = ğ¡ + 1/(2n)*sum((ğ©[i] â ğ)*(ğ©[i] â ğ)' for i=1:2n)
    # cross correlation covariance
    Pxz = 1/(2n)*sum((ğ§[i] â x)*(ğ©[i] â ğ)' for i=1:2n)
    
    # Kalman gain
    K = Pxz * Pvv^-1
    roterr = Rotations.RotationError(K * (ğ§ â ğ), CayleyMap())
    x = x â roterr
    P = P - K*Pvv*K'
    P = SMatrix{3,3}(1/2*(P+P'))
    return x, P
end

# create dataset with groundtrueth corrupted by gaussian noise
# @param Tâ discretization timestep
# @param Ïáµ¤ covariance matrix for inputs
# @param Ïáµ£ covariance matrix for measurements
function makedataset( Tâ, Ïáµ¤ , Ïáµ£ )
    # some function generating smooth rotational velocity profiles
    Ï(t) = [ 0.5*cos(sin(t))-1*cos(5*t)*sin(0.8*t)+0.25*cos(1/10*t+pi)-2*sin(0.4*t^1.5+1) ;
             0.5*sin(cos(t))-2*sin(5*t)*sin(0.8*t)+0.15*cos(1/10*t+pi/2)-0.5*sin(0.4*t^1.5+2) ; 
             0.5*cos(cos(t))-3*cos(5*t)*cos(0.8*t)+0.05*sin(1/10*t+3pi/2)-1.5*sin(0.4*t^1.5+3) ]
    # sample exact velocities for all timesteps from function
    u_trueth = hcat([ Ï(t) for t = 0:Tâ:10 ]...)
    # integrate velocities to get ground trueth rotation
    q0 = rand(UnitQuaternion)
    x_trueth = [ q0 for i=0:Tâ:10 ]
    for i=2:size(x_trueth,1)
        x_trueth[i] = UnitQuaternion(RotationVec(Tâ*u_trueth[1,i-1], Tâ*u_trueth[2,i-1], Tâ*u_trueth[3,i-1])) * x_trueth[i-1]
    end
    # distribution to corrupt rotational velocities
    q = MvNormal(zeros(3), Ïáµ¤)
    # sample random noise to add to each input
    u_noise = hcat([ u_trueth[:,i] + rand(q) for i=1:size(u_trueth,2)]...)
    # normalized measurements of the exact state
    z_trueth = hcat(Vector[[x_trueth[i]^-1 * SVector(0, 0, 1); x_trueth[i]^-1 * SVector(1, 0, 0)] for i=1:size(x_trueth,1)]...)
    # distribution to corrupt measurements
    r = MvNormal(zeros(6), Ïáµ£)
    # sample random noise to add to each meausrement
    z_noise = hcat([ z_trueth[:,i] + rand(r) for i = 1:size(z_trueth, 2)]...)
    return x_trueth, u_trueth, z_trueth, u_noise, z_noise
end

# make dataset
Ïáµ¤ = diagm(5Ï/180*ones(3))
Ïáµ£ = diagm(3Ï/180*ones(6))
x_trueth, u_trueth, z_trueth, u_noise, z_noise = makedataset( 0.001, Ïáµ¤ , Ïáµ£ )

# initialize state and covariance matrices
x = ones(UnitQuaternion)
Q = SMatrix{3,3}(Ïáµ¤)
P = Q # carful to initialize rotations "small enough"
Tâ = 0.01
R = SMatrix{6,6}(Ïáµ£)

# simulate filter 
x_sim = [ ones(UnitQuaternion) for i=1:10:10000 ]
for i = 20:10:10000
    u = SVector{3}(u_noise[:,i-1])
    global x, P = predict(x, P, u, Q, Tâ)
    z = SVector{6}(z_noise[:,i-1])
    global x, P = update(x, P, z, R)
    # two quaternions may represent the same rotation
    if x.w < 0
        global x = -x
    end
    x_sim[Int64(ceil(i/10))] = x
end

# two quaternions may represent the same rotation
for i = 1:size(x_trueth,1)
    if x_trueth[i].w < 0
        x_trueth[i] = -x_trueth[i]
    end
end

# plot quaternions
x_sim_vec = hcat([ [x_sim[i].x; x_sim[i].y; x_sim[i].z] for i=1:1000]...)
x_trueth_vec = hcat( [ [x_trueth[i].x; x_trueth[i].y; x_trueth[i].z] for i=1:size(x_trueth,1)]... )

l = @layout [a b c; d]
p1 = plot(0:0.001:10, u_noise', label = ["Ïâ" "Ïâ" "Ïâ"], title="ang velocity", xlabel="time [s]", ylabel="angular velocity [rad/s]")
p1 = plot!(0:0.001:10, u_trueth', color=:black, linewidth=1, label = ["" "" ""])
p2 = plot(0:0.001:10,z_noise[1:3, :]', label = ["gâ" "gâ" "gâ"], title="gravity vector", xlabel="time [s]", ylabel="gravity normed")
p2 = plot!(0:0.001:10, z_trueth[1:3, :]', color=:black, linewidth=1, label = ["" "" ""])
p3 = plot(0:0.001:10,z_noise[4:6, :]', label = ["nâ" "nâ" "nâ"], title="north heading", xlabel="time [s]", ylabel="north normed")
p3 = plot!(0:0.001:10, z_trueth[4:6, :]', color=:black, linewidth=1, label = ["" "" ""])
p4 = plot(0:0.01:10-0.01, x_sim_vec', color=[:red1 :green1 :blue1 ], label = ["true qx" "true qy" "true qz"], xlabel="time [s]", ylabel="quaternion vector part", title="orientation")
p4 = plot!(0:0.001:10, x_trueth_vec', color=[:red2 :green2 :blue2 ], linestyle=:dash, label = ["esitmate qx" "esitmate qy" "esitmate qz"])
plot(p1, p2, p3, p4, layout = l)