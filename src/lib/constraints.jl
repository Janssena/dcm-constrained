import Flux: sigmoid, hardsigmoid, softsign, celu, @functor

struct Initialize{T}
    init::Vector{T}
end

(i::Initialize{T})(x) where {T<:Real} = (celu.(x, 0.999) .+ 1) .* i.init
Initialize(init::Vector{T}) where T = Initialize{T}(init)


adj_sigmoid(x, lb, ub, scale) = sigmoid.(scale .* x) .* (ub .- lb) .+ lb

# Works between -1 and 1
struct TightNormalConstraint{T}
    lb::Vector{T}
    ub::Vector{T}
end

# We multiply by 6 here to push the sigmoid to operate into roughly [-1, 1]
(c::TightNormalConstraint{T})(x) where {T<:Real} = adj_sigmoid(x, c.lb, c.ub, 6.) # sigmoid.(6. .* x) .* (c.ub - c.lb) .+ c.lb
TightNormalConstraint(ub) = TightNormalConstraint{typeof(first(ub))}(zero.(ub), ub)



# Works between -3 and 3 (Do we want it to? Does that hurt performance?)
struct NormalConstraint{T}
    lb::Vector{T}
    ub::Vector{T}
end

(c::NormalConstraint{T})(x) where {T<:Real} = adj_sigmoid(x, c.lb, c.ub, 2.) #sigmoid.(2. .* x) .* (c.ub - c.lb) .+ c.lb
NormalConstraint(ub) = NormalConstraint{typeof(first(ub))}(zero.(ub), ub)

# This one is slighly more complicated: we use the regular NormalConstraint, 
# except that the user can now select one of the parameters that has a 
# propotional upper limit to one of the other parameters. These parameters are 
# chosen by using a pair: p[1] => p[2], i.e. upper limit p[2] = ratio * p[1].
struct ConditionalNormalConstraint{T}
    lb::Vector{T}
    ub::Vector{T} # will already contain the ratio
    Iₐ::Vector{Bool}
    ratio::T
    pair::Pair{Int64, Int64}
end

function ConditionalNormalConstraint(lb, ub, pair::Pair{Int64, Int64}, ratio)
    Iₐ = indicator(length(lb), pair.second)[:, 1]
    ub[pair.second] = 0.
    return ConditionalNormalConstraint{typeof(first(lb))}(lb, ub, Iₐ, ratio, pair)
end

ConditionalNormalConstraint(ub, pair::Pair{Int64, Int64}, ratio) = ConditionalNormalConstraint(zero.(ub), ub, pair, ratio)

function (c::ConditionalNormalConstraint{T})(x) where {T<:Real}
    tmp = adj_sigmoid(x, c.lb, c.ub, 2.)
    ub_cond = c.Iₐ .* c.ratio .* tmp[c.pair.first, :]'
    lb_cond = (c.Iₐ .* c.lb)[:, 1]
    return tmp + adj_sigmoid(x, lb_cond, ub_cond, 2.)
end

# Version to work with vector based x. Because of the ub_cond calculation always being a matrix the output is also a matrix.
# Necessary for making predictions for Individuals.
(c::ConditionalNormalConstraint{T})(x::AbstractVector) where {T<:Real} = c(reshape(x, length(x), 1))[:, 1]


# Alternatives are:
# hardsigmoid i.e. linear (equal likelihood to values in range)
# softsign (more likelihood to values close to mean; must be transposed) 


# Works between -1 and 1 # Very bad for performance!
struct LinearConstraint{T}
    lb::Vector{T}
    ub::Vector{T}
end

(c::LinearConstraint{T})(x) where {T<:Real} = hardsigmoid.(x) .* (c.ub - c.lb) .+ c.lb
LinearConstraint(ub) = LinearConstraint{typeof(first(ub))}(zero.(ub), ub)

# Works in a far longer interval. Makes it very hard to get extreme values.
struct SoftSignConstraint{T}
    lb::Vector{T}
    ub::Vector{T}
end

(c::SoftSignConstraint{T})(x) where {T<:Real} = (0.5 .* softsign.(6. .* x) .+ 0.5) .* (c.ub - c.lb) .+ c.lb
SoftSignConstraint(ub) = SoftSignConstraint{typeof(first(ub))}(zero.(ub), ub)


# Fixing certain parameters for all individuals:
struct AddFixedParameters{F, M}
    Iₐ_ζ::Matrix{Bool} # indicator function always returns a Matrix
    Iₐ_θ::Matrix{Bool}
    θ::M
    σ::F
end

"""constructor"""
function AddFixedParameters(idxs::Vector{Int}, p_length::Int, σ::F=identity; init::AbstractVector=[]) where {F}
    non_idxs = collect(1:p_length)
    deleteat!(non_idxs, idxs)
    I_ζ = indicator(p_length, non_idxs)
    I_θ = indicator(p_length, idxs)
    θ = length(init) > 0 ? Float32.(init) : rand(Float32, length(idxs))
    return AddFixedParameters{F, typeof(θ)}(I_ζ, I_θ, θ, σ)
end

@functor AddFixedParameters (θ,)

(fp::AddFixedParameters)(ζ::AbstractVecOrMat) = fp.σ.(fp.Iₐ_ζ * ζ .+ fp.Iₐ_θ * fp.θ)

function Base.show(io::IO, fp::AddFixedParameters{F,M}) where {F,M} 
    num_params = size(fp.Iₐ_ζ, 1)
    i = Integer(sum(fp.Iₐ_ζ))
    j = Integer(num_params)
    idxs = findall(value -> value === 1, sum(fp.Iₐ_θ, dims=2)[:, 1])
    activation = fp.σ === identity ? "" : ", $(String(F.name.name)[2:end])"
    print(io, "AddFixedParameters($i => $(j), ζ$(idxs)$(activation))")
end


# Essentially the same as AddFixedParameters but affects only θ with its σ
struct Concatenate{F, M}
    Iₐ_ζ::Matrix{Bool} # indicator function always returns a Matrix
    Iₐ_θ::Matrix{Bool}
    θ::M
    σ::F
end

"""constructor"""
function Concatenate(idxs::Vector{Int}, p_length::Int, σ::F=identity; init::AbstractVector=[]) where {F}
    non_idxs = collect(1:p_length)
    deleteat!(non_idxs, idxs)
    I_ζ = indicator(p_length, non_idxs)
    I_θ = indicator(p_length, idxs)
    θ = length(init) > 0 ? Float32.(init) : rand(Float32, length(idxs))
    return Concatenate{F, typeof(θ)}(I_ζ, I_θ, θ, σ)
end

@functor Concatenate (θ,)

(c::Concatenate)(ζ::AbstractVecOrMat) = c.Iₐ_ζ * ζ .+ c.Iₐ_θ * c.σ.(c.θ)

function Base.show(io::IO, c::Concatenate{F,M}) where {F,M} 
    num_params = size(c.Iₐ_ζ, 1)
    i = Integer(sum(c.Iₐ_ζ))
    j = Integer(num_params)
    idxs = findall(value -> value === 1, sum(c.Iₐ_θ, dims=2)[:, 1])
    activation = c.σ === identity ? "" : ", $(String(F.name.name)[2:end])"
    print(io, "Concatenate($i => $(j), ζ$(idxs)$(activation))")
end


"""Interpretable neural network"""

# Step one: take vector x or matrix X and create row-wise inputs for Parallel
struct Split{T}
    k::T # the indexes of x that go into each sub-split
end

Split() = Split(())
Split(args::Vararg{AbstractVector}) = Split(args)
(s::Split{Tuple{}})(x::AbstractVecOrMat) = tuple(split_(x)...)
(s::Split)(x::AbstractVecOrMat) = tuple(split_(x, s.k)...)

split_(x::AbstractVector) = [[i] for i in x]
split_(x::AbstractMatrix) = [transpose(x[i, :]) for i in 1:size(x, 1)]

split_(x::AbstractVector, k) = [x[k[i]] for i in eachindex(k)]
split_(x::AbstractMatrix, k) = [x[k[i], :] for i in eachindex(k)]

# Connect to special layer that takes the product 
struct Join
    Iₐ_vec::Vector{Matrix{Bool}} # information for each output to combine 
    negatives::Vector{Vector{Bool}} # the columns wise sum of the inverse of Iₐ to make sure all 0 elements are 1.
    pairs::Tuple
end

function Join(n::Int, pairs::Vararg{Pair})
    Iₐ_vec = Vector{Matrix{Bool}}(undef, length(pairs))
    negatives = Vector{Vector{Bool}}(undef, length(pairs))
    for pair in pairs
        Iₐ = indicator(n, pair.second)
        Iₐ_vec[pair.first] = Iₐ
        negatives[pair.first] = vec(sum(Iₐ, dims=2) .== 0) # sum() gives matrix
    end
    return Join(Iₐ_vec, negatives, pairs)
end

join_f(x::AbstractVecOrMat, Iₐ::Matrix{Bool}, negative::Vector{Bool}) = Iₐ * x .+ negative

function (j::Join)(x::Vararg{AbstractMatrix})
    initial = ones(length(j.negatives[1]), size(x[1])[end])
    joined = join_(initial, x, j.Iₐ_vec, j.negatives)

    return joined
end

function (j::Join)(x::Vararg{AbstractVector})
    initial = ones(length(j.negatives[1]))
    joined = join_(initial, x, j.Iₐ_vec, j.negatives)

    return joined
end

function join_(initial, x, Iₐ_vec, negatives)
    for (i, xᵢ) in enumerate(x)
        initial = initial .* join_f(xᵢ, Iₐ_vec[i], negatives[i]) # using .*= results in error mutating operation, copyto!(Matrix)
    end

    return initial
end

function Base.show(io::IO, j::Join)
    print(io, "Join($(join(["x$(pair.first) => ζ$(pair.second)" for pair in j.pairs], ", ")))")
end