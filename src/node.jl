import Random

using Lux
using ComponentArrays
using SciMLSensitivity
using DifferentialEquations

struct Normalize{T} <: Lux.AbstractExplicitLayer
    lb::AbstractVector{T}
    ub::AbstractVector{T}
end

Normalize(lb::Real, ub::Real) = Normalize([lb], [ub])
Normalize(ub::Real) = Normalize([ub])
Normalize(lb::AbstractVector, ub::AbstractVector) = Normalize{eltype(lb)}(lb, ub)
Normalize(ub::AbstractVector) = Normalize{eltype(ub)}(zero.(ub), ub)

Lux.initialparameters(rng::Random.AbstractRNG, ::Normalize) = NamedTuple()
Lux.initialstates(rng::Random.AbstractRNG, l::Normalize) = (lb=l.lb, ub=l.ub)

Lux.parameterlength(::Normalize) = 0
Lux.statelength(::Normalize) = 2 # is this correct?

function (l::Normalize)(x::AbstractArray, ps, st::NamedTuple)
    y = (((x .- st.lb) ./ (st.ub - st.lb)) .- 0.5f0) .* 2.f0 # Normalizes between -1 and 1
    # y = (x .- st.lb) ./ (st.ub - st.lb) # Normalizes between 0 and 1 
    return y, st
end


basic_tgrad(u,p,t) = zero(u)

function predict_node(node, individual, z₀, p::ComponentVector, st; tmax = maximum(individual.t), interpolate = true)
    tspan = (-0.01f0, tmax)
    p.I = 0.f0
    
    function dudt(z, p, t; st_=st.node)
        dzdt, st_ = Lux.apply(node, z, p.node, st_)
        # println(t, " ", z_ .+ p.I)
        return dzdt .+ p.I
    end
  
    ff = ODEFunction{false}(dudt, tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff, z₀, tspan, p)
    sol = solve(prob; saveat=interpolate ? empty(individual.t) : individual.t, tstops=individual.callback.condition.times, callback=individual.callback)
    return sol
end

function full_predict(encoder, neuralode, decoder, population::Population, p::ComponentVector, st)
    θ, _ = encoder(population.x, p.enc, st.enc)
    d = Integer(size(θ, 1) / 2)

    res = map(y -> zero.(y), population.y)
    for i in 1:length(population)
        individual = population[i]
        z = softplus.(θ[d+1:end, i]) .* randn(d) + θ[1:d, i] 
        z′ = predict_adjoint_node(neuralode, individual, z, p, st)
        ŷ, _ = decoder(z′, p.dec, st.dec)
        res[i] .= ŷ[1, :]
    end

    return res
end

function full_predict(encoder, neuralode, decoder, individual::Individual, p::ComponentVector, st; tmax = 72.)
    θ, _ = encoder(individual.x, p.enc, st.enc)
    d = Integer(size(θ, 1) / 2)
    z = softplus.(θ[d+1:end]) .* randn(d) + θ[1:d]
    z′ = predict_node(neuralode, individual, z, p, st; tmax)
    ŷ, _ = decoder(hcat(z′.u...), p.dec, st.dec)
    return z′.t, ŷ[1, :]
end

function predict_adjoint_node(node, individual, z₀, p::ComponentVector, st)
    tspan = (-0.01f0, maximum(individual.t))

    function dudt(z, p, t; st_=st.node)
        dzdt_, st_ = Lux.apply(node, z, p.node, st_)
        return dzdt_ .+ p.I
    end
  
    ff = ODEFunction{false}(dudt, tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff, z₀, tspan, p)
    sense = InterpolatingAdjoint(autojacvec=EnzymeVJP())
    # sense = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())
    sol = solve(prob; saveat=individual.t, sensealg=sense, tstops=individual.callback.condition.times, callback=individual.callback, verbose = false)
    return hcat(sol.u...)
end

function objective(encoder, neuralode, decoder, population, p::ComponentVector, st)
    θ, _ = encoder(population.x, p.enc, st.enc)
    d = Integer(size(θ, 1) / 2)
    SSE = 0.f0
    for i in 1:length(population)
        individual = population[i]
        z = softplus.(θ[d+1:end, i]) .* randn(Float32, d) + θ[1:d, i]
        z′ = predict_adjoint_node(neuralode, individual, z, p, st)
        ŷ, _ = decoder(z′, p.dec, st.dec)
        SSE += sum(abs2, individual.y - ŷ[1, :])
    end
    return SSE
end