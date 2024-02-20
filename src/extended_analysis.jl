import Plots
import CSV

include("src/lib/compartment_models.jl");
include("src/lib/dcm.jl");
include("src/lib/population.jl");
include("src/lib/dataset.jl");
include("src/lib/model.jl");
include("src/lib/objectives.jl");
include("src/lib/constraints.jl");
include("src/lib/ci.jl");

using StatsPlots
using DataFrames
using Statistics
using KernelDensity
using Plots.Measures

result = Bool[]
ratio = Float64[]

################################################################################
#        Higher variance in solutions over folds or training replicates?       #
################################################################################


for covariates in ["wt_ht_age", "ffm_age"]
    for neurons in [8, 32, 128]
        for model_type in ["naive", "normal-constraint", "fixed-q-v2"]
            file = "constraints-paper/data/results_$(covariates)_$(model_type)_neurons_$neurons.csv"
            df = DataFrame(CSV.File(file))
            for n in [20, 60, 120]
                df_group = groupby(df[df.n .== n, [1, 2, 8, 9, 10, 11, 12]], :replicate)
                within_fold = [std(Vector(group[1, 3:end])) for group in df_group]
                over_folds = [std([x[1, idx] for x in  df_group]) for idx in 3:7]
                push!(result, median(within_fold) < median(over_folds))
                push!(ratio, median(within_fold) / median(over_folds))
            end
        end
    end
end
sum(result .== 0) #  == 0, in all cases higher variability over folds than within folds.
1 / mean(ratio) # median variance roughly 4.5x higher over folds than within folds


################################################################################
#  Are the divergent random over the replicates or specific to some data folds? #
################################################################################

result = Vector{Int64}[]
for covariates in ["wt_ht_age", "ffm_age"]
    for neurons in [8, 32, 128]
        file = "constraints-paper/data/results_$(covariates)_naive_neurons_$neurons.csv"
        df = DataFrame(CSV.File(file))
        df_group = groupby(df[:, [1, 2, 8, 9, 10, 11, 12]], :n)
        median_rmses = [median(Matrix(group[:, 3:end])) for group in df_group]
        folds_with_div = Int64[]
        for (j, group) in enumerate(df_group)
            push!(result, sum(Matrix(group[:, 3:end]) .> 1.5*median_rmses[1], dims=2)[:, 1])
        end
    end
end

Plots.bar(sum(hcat(result...), dims=2), xticks=1:20, ylabel="Number of divergent models", xlabel="Data fold", label=nothing)



# What is special about fold 2, 3, and 8? (compared to 4-7 for example)
file = "constraints-paper/data/simulation-nhanes-chelle_additive_noise.csv"
covariates = [:Weight, :Height, :Age, :FFM]
population = load(file, covariates, S1=1/1000)
population.x .= normalize_inv(population.x', population.scale_x)'


################################################################################
#                             Covariate distributions                          #
################################################################################

idxs = [2, 3, 8]
plt1 = Plots.plot(layout = (2, 2))
n = 20
for fold in idxs
    data_split = BSON.load("constraints-paper/data/split_$(n)_$(fold).bson")
    train_population = population[data_split[:train]]
    Plots.histogram!(plt1, train_population.x', layout=(2, 2), bins=20, title=reshape(covariates, 1, length(covariates)), alpha=0.5, label="Fold $fold", xlim=[(0, 200) (50, 210) (0, 85) (0, 100)])
end 

plt2 = Plots.plot(layout = (2, 2))
n = 20
for fold in [4, 5, 6]
    data_split = BSON.load("constraints-paper/data/split_$(n)_$(fold).bson")
    train_population = population[data_split[:train]]
    Plots.histogram!(plt2, train_population.x', layout=(2, 2), bins=20, title=reshape(covariates, 1, length(covariates)), alpha=0.5, label="Fold $fold", xlim=[(0, 200) (50, 210) (0, 85) (0, 100)])
end 

Plots.plot(plt1, Plots.plot(), plt2, layout=(1, 3), size=(1200, 300))

# Maybe slightly more children?
# Hypothesis: More children -> lower V1 -> fix wrong V1 for adults with rapid redistribution

################################################################################
#               Are there differences in the concentration data?               #
################################################################################

plt1 = Plots.plot(title="Bad folds")
n = 20
for (j, bad_fold) in enumerate([2, 3, 8])
    data_split = BSON.load("constraints-paper/data/split_$(n)_$(bad_fold).bson")
    train_population = population[data_split[:train]]    
    Plots.plot!(plt1, train_population.t, train_population.y, color=:black, alpha=0.2, label=nothing)
    Plots.scatter!(plt1, train_population.t, train_population.y, color=Plots.palette(:default)[j], label=nothing)
    Plots.scatter!(plt1, [0], [-10], color=Plots.palette(:default)[j], label="Fold $bad_fold")
end 

plt2 = Plots.plot(title="Good folds")
n = 20
for (j, good_fold) in enumerate([4, 5, 6, 7])
    data_split = BSON.load("constraints-paper/data/split_$(n)_$(good_fold).bson")
    train_population = population[data_split[:train]]    
    Plots.plot!(plt2, train_population.t, train_population.y, color=:black, alpha=0.2, label=nothing)
    Plots.scatter!(plt2, train_population.t, train_population.y, color=Plots.palette(:default)[j+3], label=nothing)
    Plots.scatter!(plt2, [0], [-10], color=Plots.palette(:default)[j+3], label="Fold $good_fold")
end 

Plots.plot(plt1, plt2, layout=(1, 2), xlabel="Time (in hours)", ylabel="FVIII levels (IU/mL)", size=(800, 300), ylim=(-0.1, 2.3), bottommargin=5mm, leftmargin=5mm)


################################################################################
#                        What about initial PK parameters:                     #
################################################################################

neurons = 32
covariates = [:Weight, :Height, :Age]
population = load(file, covariates, S1=1/1000)

n = 120
fold = 2
data_split = BSON.load("constraints-paper/data/split_$(n)_$(fold).bson")

reps = 100
p = Float32[]
for r in 1:reps
    ann = Flux.Chain(
        Flux.Dense(3, neurons, Flux.swish),
        Flux.Dense(neurons, 4, Flux.softplus)
    )
    if r == 1
        p = ann(population[data_split[:train]].x)
    else
        p = hcat(p, ann(population[data_split[:train]].x))
    end
end

cl = Plots.plot(kde(p[1, :]), xlabel = "Clearance (L/h)", label=nothing)
v1 = Plots.plot(kde(p[2, :]), xlabel = "Central Volume (L)", label=nothing)
q = Plots.plot(kde(p[3, :]), xlabel = "Peripheral clearance (L/h)", label=nothing)
v2 = Plots.plot(kde(p[4, :]), xlabel = "Peripheral Volume (L)", label=nothing)

Plots.plot(cl, v1, q, v2, layout=(2,2))

# Model initializes with low V1 in all cases (all parameters initialize around 0.7)

################################################################################
#   What happens to the PK parameters when training on 'bad' vs 'good' folds   #
################################################################################

ann = Flux.Chain(
    Flux.Dense(3, neurons, Flux.swish),
    Flux.Dense(neurons, 4, Flux.softplus)
)

bad_fold = 2
good_fold = 5
num_epochs = 500
save_every = 10
save_at = collect(0:save_every:num_epochs)
save_at[1] = 1

n = 120
data_split = BSON.load("constraints-paper/data/split_$(n)_$(bad_fold).bson")
p_during_opt_n120_train = zeros(Float32, 4, length(data_split[:train]), length(save_at))
p_during_opt_n120 = zeros(Float32, 4, length(data_split[:test]), length(save_at))
ann = Flux.Chain(
    Flux.Dense(3, neurons, Flux.swish),
    Flux.Dense(neurons, 4, Flux.softplus)
)
model1 = DCM(two_comp!, ann, 2)
optimizer = Flux.ADAM(1e-2)
function saving_callback(e, l; m = model1, p1 = p_during_opt_n120, p2 = p_during_opt_n120_train)
    if e == 1 || e % save_every == 0 
        println("Epoch $e, loss = $l")
        idx = e == 1 ? 1 : Integer(e / save_every) + 1
        p1[:, :, idx] = m.re(m.weights)(population[data_split[:test]].x)
        p2[:, :, idx] = m.re(m.weights)(population[data_split[:train]].x)
    end
end
fit!(model1, population[data_split[:train]], optimizer; callback=saving_callback, epochs=num_epochs)

plt1 = Plots.plot()
[Plots.plot!(plt1, save_at, p_during_opt_n120[j, :, :]', color=Plots.palette(:default)[j], alpha=0.2, label=nothing) for j in 1:4]
plt1
plta = Plots.plot()
[Plots.plot!(plta, save_at, p_during_opt_n120_train[j, :, :]', color=Plots.palette(:default)[j], label=nothing) for j in 1:4]
plta

cov_dist_bad_120_train = Plots.plot(save_at, p_during_opt_n120_train[2, :, :]', line_z=population[data_split[:train]].x[3, :]', color=:imola, label=nothing)
cov_dist_bad_120 = Plots.plot(save_at, p_during_opt_n120[2, :, :]', line_z=population[data_split[:test]].x[3, :]', color=:imola, label=nothing)


n = 20
data_split = BSON.load("constraints-paper/data/split_$(n)_$(bad_fold).bson")
p_during_opt_n20_train = zeros(Float32, 4, length(data_split[:train]), length(save_at))
p_during_opt_n20 = zeros(Float32, 4, length(data_split[:test]), length(save_at))
ann = Flux.Chain(
    Flux.Dense(3, neurons, Flux.swish),
    Flux.Dense(neurons, 4, Flux.softplus)
)
model2 = DCM(two_comp!, ann, 2)
optimizer = Flux.ADAM(1e-2)

function saving_callback(e, l; m = model2, p1 = p_during_opt_n20, p2 = p_during_opt_n20_train)
    if e == 1 || e % save_every == 0 
        println("Epoch $e, loss = $l")
        idx = e == 1 ? 1 : Integer(e / save_every) + 1
        p1[:, :, idx] = m.re(m.weights)(population[data_split[:test]].x)
        p2[:, :, idx] = m.re(m.weights)(population[data_split[:train]].x)
    end
end
fit!(model2, population[data_split[:train]], optimizer; callback=saving_callback, epochs=num_epochs)

plt2 = Plots.plot()
[Plots.plot!(plt2, save_at, p_during_opt_n20[j, :, :]', color=Plots.palette(:default)[j], alpha=0.2, label=nothing) for j in 1:4]
plt2
pltb = Plots.plot()
[Plots.plot!(pltb, save_at, p_during_opt_n20_train[j, :, :]', color=Plots.palette(:default)[j], label=nothing) for j in 1:4]
pltb

cov_dist_bad_train = Plots.plot(save_at, p_during_opt_n20_train[2, :, :]', line_z=population[data_split[:train]].x[3, :]', color=:imola, label=nothing)
cov_dist_bad = Plots.plot(save_at, p_during_opt_n20[2, :, :]', line_z=population[data_split[:test]].x[3, :]', color=:imola, label=nothing)

n = 20
data_split = BSON.load("constraints-paper/data/split_$(n)_$(good_fold).bson")
p_during_opt_good_n20_train = zeros(Float32, 4, length(data_split[:train]), length(save_at))
p_during_opt_good_n20 = zeros(Float32, 4, length(data_split[:test]), length(save_at))
ann = Flux.Chain(
    Flux.Dense(3, neurons, Flux.swish),
    Flux.Dense(neurons, 4, Flux.softplus)
)
model3 = DCM(two_comp!, ann, 2)
optimizer = Flux.ADAM(1e-2)

function saving_callback(e, l; m = model3, p1 = p_during_opt_good_n20, p2 = p_during_opt_good_n20_train)
    if e == 1 || e % save_every == 0 
        println("Epoch $e, loss = $l")
        idx = e == 1 ? 1 : Integer(e / save_every) + 1
        p1[:, :, idx] = m.re(m.weights)(population[data_split[:test]].x)
        p2[:, :, idx] = m.re(m.weights)(population[data_split[:train]].x)
    end
end
fit!(model3, population[data_split[:train]], optimizer; callback=saving_callback, epochs=num_epochs)

plt3 = Plots.plot()
[Plots.plot!(plt3, save_at, p_during_opt_good_n20[j, :, :]', color=Plots.palette(:default)[j], alpha=0.2, label=false) for j in 1:4]
[Plots.plot!(plt3, [0, 1], [10, 10], linewidth=2, label=["CL" "V1" "Q" "V2"][j], color=Plots.palette(:default)[j], legend=:outerright) for j in 1:4]
plt3

pltc = Plots.plot()
[Plots.plot!(pltc, save_at, p_during_opt_good_n20_train[j, :, :]', color=Plots.palette(:default)[j], label=nothing) for j in 1:4]
[Plots.plot!(pltc, [0, 1], [10, 10], linewidth=2, label=["CL" "V1" "Q" "V2"][j], color=Plots.palette(:default)[j], legend=:outerright) for j in 1:4]
pltc

cov_dist_good_train = Plots.plot(save_at, p_during_opt_good_n20_train[2, :, :]', line_z=population[data_split[:train]].x[3, :]', color=:imola, label=nothing)
cov_dist_good = Plots.plot(save_at, p_during_opt_good_n20[2, :, :]', line_z=population[data_split[:test]].x[3, :]', color=:imola, label=nothing)

Plots.plot(plta, pltb, pltc, plt1, plt2, plt3, layout=Plots.grid(2, 3, widths=(0.3, 0.3, 0.4)), size=(1200, 600), ylim=(0, 3.75), title=["n = 120 'bad fold'" "n = 20 'bad fold'" "n = 20 'good fold'" "test set predictions" "test set predictions" "test set predictions"])


Plots.plot(cov_dist_bad_120_train, cov_dist_bad_train, cov_dist_good_train, cov_dist_bad_120, cov_dist_bad, cov_dist_good, layout=Plots.grid(2, 3, widths=(0.3, 0.3, 0.4)), size=(1200, 600), ylim=(0, 3.75), title=["n = 120 'bad fold'" "n = 20 'bad fold'" "n = 20 'good fold'" "test set predictions" "test set predictions" "test set predictions"], colorbar=[false false true false false true], colorbartitle="Age", ylabel=["Volume of distribution" "" "" "Volume of distribution" "" ""], leftmargin=5mm)


################################################################################
#           What happens to the PK parameters when using constraints?          #
################################################################################

n = 20
data_split = BSON.load("constraints-paper/data/split_$(n)_$(bad_fold).bson")

ann = Flux.Chain(
    Flux.Dense(3, neurons, Flux.swish),
    Flux.Dense(neurons, 4, Flux.softplus)
)

ann_init = Flux.Chain(
    Flux.Dense(3, neurons, Flux.swish),
    Flux.Dense(neurons, 4),
    Initialize([0.25, 3.35, 0.2, 1.])
)

ann_bound = Flux.Chain(
    Flux.Dense(3, neurons, Flux.swish),
    Flux.Dense(neurons, 4),
    NormalConstraint([0.0, 0.3, 0.05, 0.], [0.5, 7., 0.5, 2.])
)

ann_global = Flux.Chain(
    Flux.Dense(3, neurons, Flux.swish),
    Flux.Dense(neurons, 2),
    AddFixedParameters([3, 4], 4, Flux.softplus),
)

naive_p_during_opt_n20_train = zeros(Float32, 4, length(data_split[:train]), length(save_at))
naive_p_during_opt_n20 = zeros(Float32, 4, length(data_split[:test]), length(save_at))
model_naive = DCM(two_comp!, ann, 2)
optimizer = Flux.ADAM(1e-2)

function saving_callback(e, l; m = model_naive, p1 = naive_p_during_opt_n20, p2 = naive_p_during_opt_n20_train)
    if e == 1 || e % save_every == 0 
        println("Epoch $e, loss = $l")
        idx = e == 1 ? 1 : Integer(e / save_every) + 1
        p1[:, :, idx] = m.re(m.weights)(population[data_split[:test]].x)
        p2[:, :, idx] = m.re(m.weights)(population[data_split[:train]].x)
    end
end
fit!(model_naive, population[data_split[:train]], optimizer; callback=saving_callback, epochs=num_epochs)

plt_naive = Plots.plot()
[Plots.plot!(plt_naive, save_at, naive_p_during_opt_n20[j, :, :]', color=Plots.palette(:default)[j], alpha=0.2, label=nothing) for j in 1:4]
plt_naive

init_p_during_opt_n20_train = zeros(Float32, 4, length(data_split[:train]), length(save_at))
init_p_during_opt_n20 = zeros(Float32, 4, length(data_split[:test]), length(save_at))
model_init = DCM(two_comp!, ann_init, 2)
optimizer = Flux.ADAM(1e-2)

function saving_callback(e, l; m = model_init, p1 = init_p_during_opt_n20, p2 = init_p_during_opt_n20_train)
    if e == 1 || e % save_every == 0 
        println("Epoch $e, loss = $l")
        idx = e == 1 ? 1 : Integer(e / save_every) + 1
        p1[:, :, idx] = m.re(m.weights)(population[data_split[:test]].x)
        p2[:, :, idx] = m.re(m.weights)(population[data_split[:train]].x)
    end
end
fit!(model_init, population[data_split[:train]], optimizer; callback=saving_callback, epochs=num_epochs)

plt_init = Plots.plot()
[Plots.plot!(plt_init, save_at, init_p_during_opt_n20[j, :, :]', color=Plots.palette(:default)[j], alpha=0.2, label=nothing) for j in 1:4]
plt_init

bound_p_during_opt_n20_train = zeros(Float32, 4, length(data_split[:train]), length(save_at))
bound_p_during_opt_n20 = zeros(Float32, 4, length(data_split[:test]), length(save_at))
model_bound = DCM(two_comp!, ann_bound, 2)
optimizer = Flux.ADAM(1e-2)

function saving_callback(e, l; m = model_bound, p1 = bound_p_during_opt_n20, p2 = bound_p_during_opt_n20_train)
    if e == 1 || e % save_every == 0 
        println("Epoch $e, loss = $l")
        idx = e == 1 ? 1 : Integer(e / save_every) + 1
        p1[:, :, idx] = m.re(m.weights)(population[data_split[:test]].x)
        p2[:, :, idx] = m.re(m.weights)(population[data_split[:train]].x)
    end
end
fit!(model_bound, population[data_split[:train]], optimizer; callback=saving_callback, epochs=num_epochs)

plt_bound = Plots.plot()
[Plots.plot!(plt_bound, save_at, bound_p_during_opt_n20[j, :, :]', color=Plots.palette(:default)[j], alpha=0.2, label=nothing) for j in 1:4]
plt_bound

global_p_during_opt_n20_train = zeros(Float32, 4, length(data_split[:train]), length(save_at))
global_p_during_opt_n20 = zeros(Float32, 4, length(data_split[:test]), length(save_at))
model_global = DCM(two_comp!, ann_global, 2)
optimizer = Flux.ADAM(1e-2)

function saving_callback(e, l; m = model_global, p1 = global_p_during_opt_n20, p2 = global_p_during_opt_n20_train)
    if e == 1 || e % save_every == 0 
        println("Epoch $e, loss = $l")
        idx = e == 1 ? 1 : Integer(e / save_every) + 1
        p1[:, :, idx] = m.re(m.weights)(population[data_split[:test]].x)
        p2[:, :, idx] = m.re(m.weights)(population[data_split[:train]].x)
    end
end
fit!(model_global, population[data_split[:train]], optimizer; callback=saving_callback, epochs=num_epochs)

plt_global = Plots.plot()
[Plots.plot!(plt_global, save_at, global_p_during_opt_n20[j, :, :]', color=Plots.palette(:default)[j], alpha=0.2, label=nothing) for j in 1:4]
[Plots.plot!(plt_global, [0, 1], [10, 10], linewidth=2, label=["CL" "V1" "Q" "V2"][j], color=Plots.palette(:default)[j], legend=:outerright) for j in 1:4]
plt_global


Plots.plot(plt_naive, plt_init, plt_bound, plt_global, layout=Plots.grid(1, 4, widths=[0.21, 0.21, 0.21, 0.27]), size=(1600, 300), ylim=(0, 7), title=["naive" "initialization" "bounds" "global parameters"], bottommargin=5mm)


plt = Plots.plot(plt_naive, plt_global, layout=(1, 2), xlim=(0, 100), size=(600, 300), ylim=(0, 3.5), title=["naive" "global parameters"], xlabel="Epoch", bottommargin=5mm)


################################################################################
#               Can we use global parameters for just q or v2?                 #
################################################################################

ann_global_q = Flux.Chain(
    Flux.Dense(3, neurons, Flux.swish),
    Flux.Dense(neurons, 3),
    AddFixedParameters([3], 4, Flux.softplus),
)
global_q_p_during_opt_n20_train = zeros(Float32, 4, length(data_split[:train]), length(save_at))
global_q_p_during_opt_n20 = zeros(Float32, 4, length(data_split[:test]), length(save_at))
model_global_q = DCM(two_comp!, ann_global_v2, 2)
optimizer = Flux.ADAM(1e-2)

function saving_callback(e, l; m = model_global_q, p1 = global_q_p_during_opt_n20, p2 = global_q_p_during_opt_n20_train)
    if e == 1 || e % save_every == 0 
        println("Epoch $e, loss = $l")
        idx = e == 1 ? 1 : Integer(e / save_every) + 1
        p1[:, :, idx] = m.re(m.weights)(population[data_split[:test]].x)
        p2[:, :, idx] = m.re(m.weights)(population[data_split[:train]].x)
    end
end
fit!(model_global_q, population[data_split[:train]], optimizer; callback=saving_callback, epochs=num_epochs)

plt_global_q = Plots.plot()
[Plots.plot!(plt_global_q, save_at, global_q_p_during_opt_n20[j, :, :]', color=Plots.palette(:default)[j], alpha=0.2, label=nothing) for j in 1:4]
plt_global_q

ann_global_v2 = Flux.Chain(
    Flux.Dense(3, neurons, Flux.swish),
    Flux.Dense(neurons, 3),
    AddFixedParameters([4], 4, Flux.softplus),
)
global_v2_p_during_opt_n20_train = zeros(Float32, 4, length(data_split[:train]), length(save_at))
global_v2_p_during_opt_n20 = zeros(Float32, 4, length(data_split[:test]), length(save_at))
model_global_v2 = DCM(two_comp!, ann_global_v2, 2)
optimizer = Flux.ADAM(1e-2)

function saving_callback(e, l; m = model_global_v2, p1 = global_v2_p_during_opt_n20, p2 = global_v2_p_during_opt_n20_train)
    if e == 1 || e % save_every == 0 
        println("Epoch $e, loss = $l")
        idx = e == 1 ? 1 : Integer(e / save_every) + 1
        p1[:, :, idx] = m.re(m.weights)(population[data_split[:test]].x)
        p2[:, :, idx] = m.re(m.weights)(population[data_split[:train]].x)
    end
end
fit!(model_global_v2, population[data_split[:train]], optimizer; callback=saving_callback, epochs=num_epochs)

plt_global_v2 = Plots.plot()
[Plots.plot!(plt_global_v2, save_at, global_v2_p_during_opt_n20[j, :, :]', color=Plots.palette(:default)[j], alpha=0.2, label=nothing) for j in 1:4]
[Plots.plot!(plt_global_v2, [0, 1], [10, 10], linewidth=2, label=["CL" "V1" "Q" "V2"][j], color=Plots.palette(:default)[j], legend=:outerright) for j in 1:4]
plt_global_v2

Plots.plot(plt_global_q, plt_global_v2, layout=Plots.grid(1, 2, widths=[0.45, 0.55]), ylim=(0, 4.5), title=["Global Q" "Global V2"], size=(700, 250))



################################################################################
#    What happens when the initial value of v2 is poor in the global model?    #
################################################################################

num_epochs = 5000
save_every = 10
save_at = collect(0:save_every:num_epochs)
save_at[1] = 1

ann_global_high_v2 = Flux.Chain(
    Flux.Dense(3, neurons, Flux.swish),
    Flux.Dense(neurons, 3),
    AddFixedParameters([4], 4, Flux.softplus),
    x -> x .* Float32[1, 1, 1, 3]
)
global_high_v2_p_during_opt_n20_train = zeros(Float32, 4, length(data_split[:train]), length(save_at))
global_high_v2_p_during_opt_n20 = zeros(Float32, 4, length(data_split[:test]), length(save_at))
model_global_high_v2 = DCM(two_comp!, ann_global_high_v2, 2)
optimizer = Flux.ADAM(1e-2)

function saving_callback(e, l; m = model_global_high_v2, p1 = global_high_v2_p_during_opt_n20, p2 = global_high_v2_p_during_opt_n20_train)
    if e == 1 || e % save_every == 0 
        println("Epoch $e, loss = $l")
        idx = e == 1 ? 1 : Integer(e / save_every) + 1
        p1[:, :, idx] = m.re(m.weights)(population[data_split[:test]].x)
        p2[:, :, idx] = m.re(m.weights)(population[data_split[:train]].x)
    end
end
fit!(
    model_global_high_v2, 
    population[data_split[:train]], 
    optimizer; 
    callback=saving_callback, 
    epochs=num_epochs
)

plt_global_high_v2 = Plots.plot()
[Plots.plot!(plt_global_high_v2, save_at, global_high_v2_p_during_opt_n20[j, :, :]', color=Plots.palette(:default)[j], alpha=0.2, label=nothing) for j in 1:4]
plt_global_high_v2


ann_global_low_v2 = Flux.Chain(
Flux.Dense(3, neurons, Flux.swish),
    Flux.Dense(neurons, 3),
    AddFixedParameters([4], 4, Flux.softplus),
    x -> x .* Float32[1, 1, 1, 0.1]
)
global_low_v2_p_during_opt_n20_train = zeros(Float32, 4, length(data_split[:train]), length(save_at))
global_low_v2_p_during_opt_n20 = zeros(Float32, 4, length(data_split[:test]), length(save_at))
model_global_low_v2 = DCM(two_comp!, ann_global_low_v2, 2)
optimizer = Flux.ADAM(1e-2)

function saving_callback(e, l; m = model_global_low_v2, p1 = global_low_v2_p_during_opt_n20, p2 = global_low_v2_p_during_opt_n20_train)
    if e == 1 || e % save_every == 0 
        println("Epoch $e, loss = $l")
        idx = e == 1 ? 1 : Integer(e / save_every) + 1
        p1[:, :, idx] = m.re(m.weights)(population[data_split[:test]].x)
        p2[:, :, idx] = m.re(m.weights)(population[data_split[:train]].x)
    end
end
fit!(
    model_global_low_v2, 
    population[data_split[:train]], 
    optimizer; 
    callback=saving_callback, 
    epochs=num_epochs
)

plt_global_low_v2 = Plots.plot()
[Plots.plot!(plt_global_low_v2, save_at, global_low_v2_p_during_opt_n20[j, :, :]', color=Plots.palette(:default)[j], alpha=0.2, label=nothing) for j in 1:4]
[Plots.plot!(plt_global_low_v2, [0, 1], [10, 10], linewidth=2, label=["CL" "V1" "Q" "V2"][j], color=Plots.palette(:default)[j], legend=:outerright) for j in 1:4]
plt_global_low_v2

Plots.plot(plt_global_high_v2, plt_global_low_v2, xlim=(0, 1000), ylim=(0, 6), size=(700, 250), layout=Plots.grid(1, 2, widths=[0.45, 0.55]), title=["Global V2 (init at 2.7 L)" "Global V2 (init at 0.07 L)"])



################################################################################
#                     Does placing a bound on just v2 help?                    #
################################################################################

num_epochs = 500
save_every = 10
save_at = collect(0:save_every:num_epochs)
save_at[1] = 1

ann_bound_v2 = Flux.Chain(
    Flux.Dense(3, neurons, Flux.swish),
    Flux.Dense(neurons, 4),
    x -> [
        Flux.softplus.(x[1:1, :]); 
        Flux.softplus.(x[2:2, :]); 
        Flux.softplus.(x[3:3, :]); 
        NormalConstraint([2.f0])(x[4:4, :])
    ]
)
bound_v2_p_during_opt_n20_train = zeros(Float32, 4, length(data_split[:train]), length(save_at))
bound_v2_p_during_opt_n20 = zeros(Float32, 4, length(data_split[:test]), length(save_at))
model_bound_v2 = DCM(two_comp!, ann_bound_v2, 2)
optimizer = Flux.ADAM(1e-2)

function saving_callback(e, l; m = model_bound_v2, p1 = bound_v2_p_during_opt_n20, p2 = bound_v2_p_during_opt_n20_train)
    if e == 1 || e % save_every == 0 
        println("Epoch $e, loss = $l")
        idx = e == 1 ? 1 : Integer(e / save_every) + 1
        p1[:, :, idx] = m.re(m.weights)(population[data_split[:test]].x)
        p2[:, :, idx] = m.re(m.weights)(population[data_split[:train]].x)
    end
end
fit!(
    model_bound_v2, 
    population[data_split[:train]], 
    optimizer; 
    callback=saving_callback, 
    epochs=num_epochs
)

plt_bound_v2 = Plots.plot(title = "Bound on V2 only")
[Plots.plot!(plt_bound_v2, save_at, bound_v2_p_during_opt_n20[j, :, :]', color=Plots.palette(:default)[j], alpha=0.2, label=nothing) for j in 1:4]
plt_bound_v2

# softer bound:
num_epochs = 500
save_every = 10
save_at = collect(0:save_every:num_epochs)
save_at[1] = 1

ann_soft_bound_v2 = Flux.Chain(
    Flux.Dense(3, neurons, Flux.swish),
    Flux.Dense(neurons, 4),
    x -> [
        Flux.softplus.(x[1:1, :]); 
        Flux.softplus.(x[2:2, :]); 
        Flux.softplus.(x[3:3, :]); 
        SoftSignConstraint([2.f0])(x[4:4, :])
    ]
)
soft_bound_v2_p_during_opt_n20_train = zeros(Float32, 4, length(data_split[:train]), length(save_at))
soft_bound_v2_p_during_opt_n20 = zeros(Float32, 4, length(data_split[:test]), length(save_at))
model_soft_bound_v2 = DCM(two_comp!, ann_soft_bound_v2, 2)
optimizer = Flux.ADAM(1e-2)

function saving_callback(e, l; m = model_soft_bound_v2, p1 = soft_bound_v2_p_during_opt_n20, p2 = soft_bound_v2_p_during_opt_n20_train)
    if e == 1 || e % save_every == 0 
        println("Epoch $e, loss = $l")
        idx = e == 1 ? 1 : Integer(e / save_every) + 1
        p1[:, :, idx] = m.re(m.weights)(population[data_split[:test]].x)
        p2[:, :, idx] = m.re(m.weights)(population[data_split[:train]].x)
    end
end
fit!(
    model_soft_bound_v2, 
    population[data_split[:train]], 
    optimizer; 
    callback=saving_callback, 
    epochs=num_epochs
)

plt_soft_bound_v2 = Plots.plot(title = "Softer bound on V2")
[Plots.plot!(plt_soft_bound_v2, save_at, soft_bound_v2_p_during_opt_n20[j, :, :]', color=Plots.palette(:default)[j], alpha=0.2, label=nothing) for j in 1:4]
[Plots.plot!(plt_soft_bound_v2, [0, 1], [10, 10], linewidth=2, label=["CL" "V1" "Q" "V2"][j], color=Plots.palette(:default)[j], legend=:outerright) for j in 1:4]
plt_soft_bound_v2

Plots.plot(plt_bound_v2, plt_soft_bound_v2, ylim=(0, 4), size=(700, 250), layout=Plots.grid(1, 2, widths=[0.45, 0.55]))


################################################################################
#       Does removing specific samples from the data improve solutions?         #
################################################################################

n = 20
data_split = BSON.load("constraints-paper/data/split_$(n)_$(bad_fold).bson")

ann = Flux.Chain(
    Flux.Dense(3, neurons, Flux.swish),
    Flux.Dense(neurons, 4, Flux.softplus)
)

# Does optimization improve when we remove the youngest patients?
mask = .!(normalize_inv(population[data_split[:train]].x', population.scale_x)[:, 3] .< 10)

naive_mask_p_during_opt_n20_train = zeros(Float32, 4, length(data_split[:train]), length(save_at))
naive_mask_p_during_opt_n20 = zeros(Float32, 4, length(data_split[:test]), length(save_at))
model_naive_mask = DCM(two_comp!, ann, 2)
optimizer = Flux.ADAM(1e-2)

function saving_callback(e, l; m = model_naive_mask, p1 = naive_mask_p_during_opt_n20, p2 = naive_mask_p_during_opt_n20_train)
    if e == 1 || e % save_every == 0 
        println("Epoch $e, loss = $l")
        idx = e == 1 ? 1 : Integer(e / save_every) + 1
        p1[:, :, idx] = m.re(m.weights)(population[data_split[:test]].x)
        p2[:, :, idx] = m.re(m.weights)(population[data_split[:train]].x)
    end
end
fit!(model_naive_mask, population[data_split[:train][mask]], optimizer; callback=saving_callback, epochs=num_epochs)

plt_naive_mask = Plots.plot(title = "Without patients <10 years old")
[Plots.plot!(plt_naive_mask, save_at, naive_mask_p_during_opt_n20[j, :, :]', color=Plots.palette(:default)[j], alpha=0.2, label=nothing) for j in 1:4]
[Plots.plot!(plt_naive_mask, [0, 1], [10, 10], linewidth=2, label=["CL" "V1" "Q" "V2"][j], color=Plots.palette(:default)[j], legend=:outerright) for j in 1:4]
Plots.plot(plt_naive_mask, ylim=(0, 4.7))




################################################################################
#               Run V2 specific constraints on synthetic data set               #
################################################################################

file = "constraints-paper/data/simulation-nhanes-chelle_additive_noise.csv"
covariates = "ffm-age"
if covariates == "wt-ht-age"
    population = load(file, [:Weight, :Height, :Age], S1=1/1000)
else
    population = load(file, [:FFM, :Age], S1=1/1000)
end

n = 20
neurons = 8
for model_type in ["v2-bound", "v2-global"]
    for replicate in 1:20
        println("Running for n = $n, neurons = $neurons, model = $model_type, replicate $replicate")
        data_split = BSON.load("constraints-paper/data/split_$(n)_$(replicate).bson")
        train = data_split[:train]
        test = data_split[:test]
        
        train_population = population[train]
        test_population = population[test]
        
        Threads.@threads for j in ["a", "b", "c", "d", "e"]
            checkpoint_file = "constraints-paper/checkpoints/$(model_type)/$(covariates)/$(neurons)/$(n)/checkpoint_set_$(replicate)$(j).bson"
            if isfile(checkpoint_file)
                println("[ALREADY EXISTS] replicate $j")
                continue
            end
            println("replicate $j")
            if model_type === "v2-bound"
                ann = Flux.Chain(
                    Flux.Dense(covariates == "wt-ht-age" ? 3 : 2, neurons, Flux.swish),
                    Flux.Dense(neurons, 4),
                    x -> [
                        Flux.softplus.(x[1:1, :]); 
                        Flux.softplus.(x[2:2, :]); 
                        Flux.softplus.(x[3:3, :]); 
                        SoftSignConstraint([2.f0])(x[4:4, :])
                    ]
                )
            else
                ann = Flux.Chain(
                    Flux.Dense(covariates == "wt-ht-age" ? 3 : 2, neurons, Flux.swish),
                    Flux.Dense(neurons, 3),
                    AddFixedParameters([4], 4, Flux.softplus)
                )
            end
            ##### For training MSE DCM:
            model = DCM(two_comp!, ann, 2)
            optimizer = Flux.ADAM(1e-2)
            fit!(model, train_population, optimizer; epochs=500)
            
            BSON.bson(checkpoint_file, Dict(:model => model, :train => train, :test => test, :test_acc => sqrt(mse(model, test_population)), :optimizer => optimizer))
        end
    end
end

# Gather results:
for covariates in ["wt-ht-age", "ffm-age"]
    for model_type in ["v2-bound", "v2-global"]
        folder = "constraints-paper/checkpoints/$(model_type)/$(covariates)/$(neurons)/$(n)/"
        files = readdir(folder)
        result = zeros(length(files))
        for (i, file) in enumerate(files)
            result[i] = BSON.parse(joinpath(folder, file))[:test_acc]
        end
        println("($model_type using $covariates): $(median(result)) +- $(std(result)) ($(sum(result .> 1.5 * median(result)) / length(result) * 100 )%)")
    end
end