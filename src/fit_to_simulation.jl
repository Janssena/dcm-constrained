import Plots
import BSON
import Flux

using DeepCompartmentModels

include("src/lib/compartment_models.jl");
include("src/lib/dcm.jl");
include("src/lib/population.jl");
include("src/lib/dataset.jl");
include("src/lib/model.jl");
include("src/lib/objectives.jl");
include("src/lib/constraints.jl");
include("src/lib/ci.jl");

"""
##### Constraints paper:

It might be difficult for the model to find the perfect model when passed weight, 
height, and age. 

✓ We might consider also fitting a model using ffm and age, to see
if in those cases the IIV is estimated more accurately [DONE]. 

✓ We are currently using a quite wide neural network, maybe we can also try a 
thinner NN? 32 (✓) vs 124 (✓) neurons

* We want to know approaches that eliminate strange things with respect to the 
predicted parameters. Since the ODE is still quite flexible, we could identify
parameters that have higher uncertainty and thus are still flexible using 
something like Dropout (×) or Deep Ensembles (✓, we can create ensembles out of 
the replicates). The parameters with high variance are likely problematic and 
thus might require regularization. It would be nice to add this.

* Maybe doing loss landscape visualization could be 
quite cool to also evaluate the constraints? This could go in the appendix as 
supporting data, as it should not be a main outcome.


##### Mixed effects paper:

* We should use the same data set as in the constraints paper. We should then 
evaluate the best constraints as presented in the constraints paper.

* Another option might be to perform mini-batch SGD? This should make learning 
more stochastic but forces the model to (hopefully) make parameter updates that 
will also benefit the samples not in the current batch. This can be helpful to 
speed up the training of larger data sets using mixed effects models.


##### Final clinical paper:

* Through the interpretable neural network, using only five neurons and the ReLu 
is already quite decent. This approach is nice since we can visualize the 
functions and see what is happening during training. This approach should be 
shown next to a regular NME model.
"""

# Load data:
file = "constraints-paper/data/simulation-nhanes-chelle_additive.csv"
# population = load(file, [:Weight, :Height, :Age], S1=1/1000)
population = load(file, [:FFM, :Age, :BGO], S1=1/1000)

# ANN:
ann = Flux.Chain(
    Flux.Dense(3, 128, Flux.swish),
    Flux.Dense(128, 4, Flux.softplus)
)

ann_fixed = Flux.Chain(
    Flux.Dense(3, 128, Flux.swish),
    Flux.Dense(128, 2),
    AddFixedParameters([3, 4], 4, Flux.softplus),
)

ann_constrained = Flux.Chain(
    Flux.Dense(2, 128, Flux.swish),
    Flux.Dense(128, 4),
    NormalConstraint([0.0, 0.3, 0.05, 0.], [0.5, 7., 0.5, 2.]) # When using output constraint
)

ann_fixed_constrained = Flux.Chain(
    Flux.Dense(3, 128, Flux.swish),
    Flux.Dense(128, 2),
    AddFixedParameters([3, 4], 4; init=zeros(2)),
    NormalConstraint([0.0, 0.3, 0.05, 0.], [0.5, 7., 0.5, 2.])
)

Ω_init = [0.1 0.01; 0.01 0.1]
σ²_init = [0.1^2]

# Models:
model = DCM(two_comp!, ann, 2)
model_FO = DCM(FO(), two_comp!, ann, [1, 2], Ω_init, σ²_init, 2)
model_FOCE = DCM(FOCE(), two_comp!, ann, [1, 2], Ω_init, σ²_init, 2)
model_FOCEI = DCM(FOCEI(), two_comp!, ann, [1, 2], Ω_init, σ²_init, 2)
model_VI = DCM(VI(), two_comp!, ann, [1, 2], Ω_init, σ²_init, 2)

# Prepare the population to the number of random effect parameters:
population = adapt(population, model_FO)

# create train and test splits
for n in [20, 60, 120]
    for replicate in 6:20
        idxs = collect(1:length(population))
        shuffle!(idxs)
        train = idxs[1:n]
        test = idxs[n+1:end]
        @assert length(train) + length(test) == length(population) "Combined size of train and test set does not match population size"
        BSON.bson("constraints-paper/data/split_$(n)_$(replicate).bson", Dict(:train => train, :test => test))
    end
end

replicate = 1
n = 20 # size of train set
data_split = BSON.load("constraints-paper/data/split_$(n)_$(replicate).bson")
train = data_split[:train]
test = data_split[:test]

train_population = population[train]
test_population = population[test]

cb(m, trp, tep) = function f1(e, l) 
    println("Epoch $e, train loss: $(l), test loss: $(objective(m, tep))")
    display(Plots.plot(m, trp))
end

cb_mix(m, trp, tep) = function f2(e, l) 
    println("Epoch $e, train loss: $(l), test loss: $(objective(m, tep)), Ω: $(var(m)[1]), σ: $(sqrt.(var(m)[2]))")
    display(Plots.plot(m, trp))
end

################################################################################
#####                        Fit DCM based on MSE                          #####
################################################################################
"""500 epochs seems to work fine for unconstrained DCM"""
optimizer = Flux.ADAM(1e-2)
fit!(model, train_population, optimizer; callback=cb(model, train_population, test_population), epochs=500)

i = rand(1:length(test_population))
Plots.plot(predict(model, test_population[i]; interpolate=true))
Plots.plot!(test_population[i])


################################################################################
#####                        Fit DCM based on FO                           #####
################################################################################
"""
Fitting this is a little bit more complex. In our previous implementation (see 
'old' folder) the FOCE method seemed to fit really fast. Need to check the 
differences. If we start from a bad position, the IIV takes over and remains 
high for pretty long. Constraining might work, but we can also first fit using 
MSE, or have an adaptive learning rate for the IIV. For example increase 
learning rate after x epochs. A reasonable initial value for Σ might also help 
(the current initial estimate for proportional error is very low).
"""
optimizer_w = Flux.ADAM(1e-2)
optimizer_θ = Flux.ADAM(1e-2) # From 1e-3 -> 1e-2
fit!(model_FO, train_population, optimizer_w, optimizer_θ; callback=cb_mix(model_FO, train_population, test_population), epochs=500)
optimizer_θ.eta = 1e-2
fit!(model_FO, train_population, optimizer_w, optimizer_θ; callback=cb_mix(model_FO, train_population, test_population), epochs=400)

# Or maybe 400 1e-2 into 100 1e-3?

"""
* What do we consider convergence? 
Maybe from some point on we should start saving the parameters and create some 
sort of running average, i.e. we store the top 20 LL parameters?
"""

i = rand(1:length(test_population))
Plots.plot(predict(model_FO, test_population[i]; typical=true, interpolate=true), color=:dodgerblue, linewidth=2)
res = ci(model_FO, test_population[i]; Σ=var(model_FO)[1])
Plots.plot!(res.x, res.y, ribbon=res.ribbon; alpha=0, fillalpha=0.2, color=:dodgerblue)
Plots.plot!(test_population[i])

################################################################################
#####                       Fit DCM based on FOCE                          #####
################################################################################
optimizer_w = Flux.ADAM(1e-2)
optimizer_θ = Flux.ADAM(1e-3)
fit!(model_FOCE, train_population, optimizer_w, optimizer_θ; callback=cb(model_FOCE), epochs=500)



################################################################################
#####                       Fit DCM based on FOCEI                         #####
################################################################################
optimizer_w = Flux.ADAM(1e-2)
optimizer_θ = Flux.ADAM(1e-3)
fit!(model_FOCEI, train_population, optimizer_w, optimizer_θ; callback=cb_mix(model_FOCEI), epochs=500)

i = rand(1:length(test_population))
Plots.plot(predict(model_FOCEI, test_population[i]; typical=true, interpolate=true), color=:dodgerblue, linewidth=2)
res = ci(model_FOCEI, test_population[i]; Σ=var(model_FOCEI)[1])
Plots.plot!(res.x, res.y, ribbon=res.ribbon; alpha=0, fillalpha=0.2, color=:dodgerblue)
Plots.plot!(test_population[i])


"""
Perhaps good diagnostic plots would be:
* Residual error plotted over time + residual error estimate.

* Also be of interest would be to add the distance to the 9x% credible interval from the IIV.
This plot is more complex to create however. One of the questions would be if the etas
at for example the 90% CI boundary result in a comparable proportional increase 
in the predicted IVR when accounting for the dose. Maybe it does?
"""

# Collect results:
cb(m, trp, tep) = function f1(e, l) 
    println("Epoch $e, train loss: $(l), test loss: $(objective(m, tep))")
    display(Plots.plot(model, trp))
end


# We should also collect all thetas to see how they behave for each model
cb_res(m, tep, res, train_loss, test_loss) = function f1(e, l) 
    if train_loss[e] == 0.
        train_loss[e] = l
        test_loss[e] = objective(m, tep)
    else
        train_loss[e + 100] = l # if the value is already filled it means we have an adaptive lr.
        test_loss[e + 100] = objective(m, tep)
    end
    # print("Epoch $e, train loss: $(l), test loss: $(objective(m, tep))")
    if l < maximum(res.LL)
        res[res.LL .== maximum(res.LL), :LL] .= l
        res[res.LL .== maximum(res.LL), :w] .= [m.weights]
        res[res.LL .== maximum(res.LL), :theta] .= [m.theta]
        # println(" [SAVED]")
    else
        # println()
    end
end


file = "constraints-paper/data/simulation-nhanes-chelle_additive_noise.csv"
# population = load(file, [:FFM, :Age], S1=1/1000)
# DATASET #1
population = load(file, [:Weight, :Height, :Age], S1=1/1000)
covariates = "wt-ht-age"
# DATASET #2
# population = load(file, [:FFM, :Age], S1=1/1000)
# covariates = "ffm-age"


# file = "constraints-paper/data/simulation-nhanes-chelle_additive_noise.csv"
# df_true = DataFrame(CSV.File(file))
# p_true = [Vector(group[1, [:CL, :V1, :Q, :V2]]) for group in groupby(df_true, :ID)]

# prob = ODEProblem(two_comp!, zeros(2), (-0.1, 72.))
# t_ = collect(5/60:5/60:72.)
# y = zeros(length(t_), length(population))
# for i in 1:length(population)
#     prob_i = remake(prob, p=vcat(p_true[i], 0.))
#     y[:, i] = solve(prob_i, saveat=t_, save_idxs=1, tstops=callback=population.callbacks[i].condition.times, callback=population.callbacks[i]).u
# end


# Experiment 1
for n in [20, 60, 120]
    # for neurons in [8, 128]
    for neurons in [8, 32, 128]
        for model_type in ["naive", "initialization", "normal-constraint", "fixed-q-v2"]
            # filename = "constraints-paper/data/results_ffm_age_interpretable-noise-fullycon_neurons_$(neurons).csv" 
            # if isfile(filename)
            #     println("EXISTS, MOVING ON ...")
            #     continue
            # end
            # df = DataFrame(n = vcat(repeat([120 60 20], 20, 1)...), replicate = repeat(1:20, 3), train_rmse_a_72 = 0., train_rmse_b_72 = 0., train_rmse_c_72 = 0., train_rmse_d_72 = 0., train_rmse_e_72 = 0., test_rmse_a_72 = 0., test_rmse_b_72 = 0., test_rmse_c_72 = 0., test_rmse_d_72 = 0., test_rmse_e_72 = 0., train_rmse_a_6 = 0., train_rmse_b_6 = 0., train_rmse_c_6 = 0., train_rmse_d_6 = 0., train_rmse_e_6 = 0., test_rmse_a_6 = 0., test_rmse_b_6 = 0., test_rmse_c_6 = 0., test_rmse_d_6 = 0., test_rmse_e_6 = 0., train_rmse_a_6_72 = 0., train_rmse_b_6_72 = 0., train_rmse_c_6_72 = 0., train_rmse_d_6_72 = 0., train_rmse_e_6_72 = 0., test_rmse_a_6_72 = 0., test_rmse_b_6_72 = 0., test_rmse_c_6_72 = 0., test_rmse_d_6_72 = 0., test_rmse_e_6_72 = 0.)
            for replicate in 1:20
                println("Running for n = $n, neurons = $neurons, model = $model_type, replicate $replicate")
                data_split = BSON.load("constraints-paper/data/split_$(n)_$(replicate).bson")
                train = data_split[:train]
                test = data_split[:test]
                
                train_population = population[train]
                test_population = population[test]
                
                Threads.@threads for j in ["a", "b", "c", "d", "e"]
                    checkpoint_file = "constraints-paper/checkpoints/$(model_type)/$(covariates)/$(neurons)/$(n)/checkpoint_set_$(replicate)$(j).bson"
                    # checkpoint_file = "constraints-paper/checkpoints/fixed-q-v2-noise-fullycon/$(n)/checkpoint_set_$(replicate)$(j).bson"
                    # checkpoint_file = "constraints-paper/checkpoints/interpretable-noise-fullycon/$(n)/checkpoint_set_$(replicate)$(j).bson"
                    # checkpoint_file = "constraints-paper/checkpoints/mse/additive-error/wt-ht-age/$(n)/$(neurons)/$(model_type)/checkpoint_set_$(replicate)$(j).bson"
                    # checkpoint_file = "constraints-paper/checkpoints/mse/additive-error/ffm-age/$(n)/$(neurons)/$(model_type)/checkpoint_set_$(replicate)$(j).bson"
                    # checkpoint_file = "/media/alexanderjanssen/T7 Touch/backup constraints paper checkpoints/additive-error/ffm-age/$(n)/$(neurons)/$(model_type)/checkpoint_set_$(replicate)$(j).bson"
                    if isfile(checkpoint_file)
                        println("[ALREADY EXISTS] replicate $j")
                        continue
                    end
                    println("replicate $j")
                    if model_type === "naive"
                        ann = Flux.Chain(
                            Flux.Dense(2, neurons, Flux.swish),
                            Flux.Dense(neurons, 4, Flux.softplus)
                        )
                    elseif model_type === "initialization"
                        ann = Flux.Chain(
                            Flux.Dense(3, neurons, Flux.swish),
                            Flux.Dense(neurons, 4),
                            Initialize([0.25, 3.35, 0.2, 1.])
                        )
                    elseif model_type === "normal-constraint"
                        ann = Flux.Chain(
                            Flux.Dense(2, neurons, Flux.swish),
                            Flux.Dense(neurons, 4),
                            NormalConstraint([0.0, 0.3, 0.05, 0.], [0.5, 7., 0.5, 2.])
                        )
                    elseif model_type === "fixed-q-v2"
                        ann = Flux.Chain(
                            Flux.Dense(2, neurons, Flux.swish),
                            Flux.Dense(neurons, 2),
                            AddFixedParameters([3, 4], 4, Flux.softplus),
                        )
                    elseif model_type === "initialization-fixed-q-v2"
                        ann = Flux.Chain(
                            Flux.Dense(3, neurons, Flux.swish),
                            Flux.Dense(neurons, 2),
                            AddFixedParameters([3, 4], 4; init=zeros(2)),
                            Initialize([0.25, 3.35, 0.2, 1.])
                        )
                    elseif model_type === "interpretable"
                        nn_wt = Flux.Chain(Flux.Dense(1, neurons, Flux.swish), Flux.Dense(neurons, 2, Flux.softplus))
                        nn_age = Flux.Chain(Flux.Dense(1, neurons, Flux.swish), Flux.Dense(neurons, 2, Flux.softplus))
                        ann = Flux.Chain(
                            Split(),
                            Flux.Parallel(Join(2, 1 => [1, 2], 2 => [1, 2]), nn_wt, nn_age),
                            Concatenate([3, 4], 4, Flux.softplus)
                        )
                    elseif model_type === "interpretable-ageonv1"
                        neurons_ = 32
                        nn_ffm = Flux.Chain(Flux.Dense(1, neurons_, Flux.swish), Flux.Dense(neurons_, 2, Flux.softplus))
                        nn_age = Flux.Chain(Flux.Dense(1, neurons_, Flux.swish), Flux.Dense(neurons_, 1, Flux.softplus))
                        ann = Flux.Chain(
                            Split(),
                            Flux.Parallel(Join(2, 1 => [1, 2], 2 => [1]), nn_ffm, nn_age), # note: not having age affect v1
                            Concatenate([3], 3, (x) -> Flux.sigmoid(x) * 0.4),
                            Concatenate([4], 4, (x) -> Flux.sigmoid(x) * 1.0),
                        )
                        
                        # nn_wt = Flux.Chain(Flux.Dense(1, neurons, Flux.swish), Flux.Dense(neurons, 2, Flux.softplus))
                        # nn_age = Flux.Chain(Flux.Dense(1, neurons, Flux.swish), Flux.Dense(neurons, 1, Flux.softplus))
                        # ann = Flux.Chain(
                        #     Split(),
                        #     Flux.Parallel(Join(2, 1 => [1, 2], 2 => [2]), nn_wt, nn_age), # note: not having age affect v1
                        #     Concatenate([3, 4], 4, Flux.softplus)
                        # )
                    elseif model_type === "interpretable-noise"
                        neurons_ = 16
                        nn_ffm = Flux.Chain(Flux.Dense(1, neurons_, Flux.swish), Flux.Dense(neurons_, 2, Flux.softplus; bias=[0., 0.5]))
                        nn_age = Flux.Chain(Flux.Dense(1, neurons_, Flux.swish), Flux.Dense(neurons_, 2, Flux.softplus; bias=[0., 0.5]))
                        nn_noise = Flux.Chain(Flux.Dense(1, neurons_, Flux.swish), Flux.Dense(neurons_, 2, Flux.softplus; bias=[0., 0.5]))
                        nn_noise2 = Flux.Chain(Flux.Dense(1, neurons_, Flux.swish), Flux.Dense(neurons_, 2, Flux.softplus; bias=[0., 0.5]))
                        nn_catnoise = Flux.Chain(Flux.Dense(1, neurons_, Flux.swish), Flux.Dense(neurons_, 2, Flux.softplus; bias=[0., 0.5]))
                        ann = Flux.Chain(
                            Split(),
                            Flux.Parallel(Join(2, 1 => [1, 2], 2 => [1, 2], 3 => [1, 2], 4 => [1, 2], 5 => [1, 2]), nn_ffm, nn_age, nn_noise, nn_noise2, nn_catnoise),
                            Concatenate([3, 4], 4, Flux.softplus)
                        )
                    elseif model_type === "conditional-normal-constraint"
                        ann = Flux.Chain(
                            Flux.Dense(3, neurons, Flux.swish),
                            Flux.Dense(neurons, 4),
                            ConditionalNormalConstraint([0.0, 0.3, 0.05, 0.], [0.5, 7., 0., 2.], 2 => 3, 0.33) # this constrains k₁₂ ≤ 0.33
                        )
                    elseif model_type === "tight-normal-constraint"
                        ann = Flux.Chain(
                            Flux.Dense(3, neurons, Flux.swish),
                            Flux.Dense(neurons, 4),
                            TightNormalConstraint([0.0, 0.3, 0.05, 0.], [0.5, 7., 0.5, 2.])
                        )
                    elseif model_type === "linear-constraint"
                        ann = Flux.Chain(
                            Flux.Dense(2, neurons, Flux.swish),
                            Flux.Dense(neurons, 4),
                            LinearConstraint([0.0, 0.3, 0.05, 0.01], [0.5, 7., 0.5, 2.])
                        )
                    elseif model_type === "softsign-constraint"
                        ann = Flux.Chain(
                            Flux.Dense(2, neurons, Flux.swish),
                            Flux.Dense(neurons, 4),
                            SoftSignConstraint([0.0, 0.3, 0.05, 0.], [0.5, 7., 0.5, 2.])
                        )
                    else
                        ann = Flux.Chain(
                            Flux.Dense(2, neurons, Flux.swish),
                            Flux.Dense(neurons, 2),
                            AddFixedParameters([3, 4], 4; init=zeros(2)),
                            NormalConstraint([0.0, 0.3, 0.05, 0.], [0.5, 7., 0.5, 2.])
                        )
                    end
                    ##### For training MSE DCM:
                    model = DCM(two_comp!, ann, 2)
                    optimizer = Flux.ADAM(1e-2)
                    fit!(model, train_population, optimizer; epochs=500)
                    # fit!(model, train_population, optimizer; epochs=500, callback=cb(model, train_population, test_population))
                    
                    # BSON.bson(checkpoint_file, Dict(:model => model, :population => population, :train => train, :test => test, :optimizer => optimizer))
                    
                    # predictions = hcat(map(sol -> sol(t_).u, predict(model, population; tmax=72., interpolate=true))...)
                    # squared_errors = (y - predictions).^2
                    # idx = first((1:nrow(df))[(df.n .== n) .& (df.replicate .== replicate)])
                    # df[idx, "train_rmse_$(j)_72"] = sqrt(mean(mean(squared_errors[:, train], dims=2)))
                    # df[idx, "test_rmse_$(j)_72"] = sqrt(mean(mean(squared_errors[:, test], dims=2)))
                    # df[idx, "train_rmse_$(j)_6"] = sqrt(mean(mean(squared_errors[1:length(5/60:5/60:6.), train], dims=2)))
                    # df[idx, "test_rmse_$(j)_6"] = sqrt(mean(mean(squared_errors[1:length(5/60:5/60:6.), test], dims=2)))
                    # df[idx, "train_rmse_$(j)_6_72"] = sqrt(mean(mean(squared_errors[length(5/60:5/60:6.):end, train], dims=2)))
                    # df[idx, "test_rmse_$(j)_6_72"] = sqrt(mean(mean(squared_errors[length(5/60:5/60:6.):end, test], dims=2)))

                    BSON.bson(checkpoint_file, Dict(:model => model, :train => train, :test => test, :test_acc => sqrt(mse(model, test_population)), :optimizer => optimizer))
                    # Note: The digit in the file name refers to the replicate of the train test split.,
                    # The letter directly following referes to replicate of the random 
                    # initialization of the neural network.
                    
                    ##### For training FO NME
                    # train_loss = zeros(500)
                    # test_loss = zeros(500)
                    # model = DCM(FO(), two_comp!, ann, [1, 2], Ω_init, σ²_init, 2)
                    # result = DataFrame(LL = rand(20), w = fill(model.weights, 20), theta = fill(model.theta, 20))
                    # optimizer_w = Flux.ADAM(1e-2)
                    # optimizer_θ = Flux.ADAM(1e-3) # From 1e-3 -> 1e-2
                    # fit!(model, train_population, optimizer_w, optimizer_θ; callback=cb_res(model, test_population, result, train_loss, test_loss), epochs=100)
                    # optimizer_θ.eta = 1e-2 # From 1e-3 -> 1e-2
                    # fit!(model, train_population, optimizer_w, optimizer_θ; callback=cb_res(model, test_population, result, train_loss, test_loss), epochs=400)

                    # BSON.bson("constraints-paper/checkpoints/FO/additive-error/adaptive-lr/wt-ht-age/$(n)/$(neurons)/$(model_type)/checkpoint_set_$(replicate)$(j).bson", Dict(:model => model, :population => population, :train => train, :test => test, :optimizer => [optimizer_w, optimizer_θ], :best_20 => result, :train_loss => train_loss, :test_loss => test_loss))
                    

                    # Checkpointing like this unfortunately does not make the 
                    # model usable. The functions inside the callbacks cannnot 
                    # be used like this. We do save the population however so 
                    # that we can check if we use the correct population.
                end
            end
            # CSV.write(filename, df)
        end
    end
end



# DATASET #3
population = load(file, [:FFM, :Age, :Noise, :Noise2, :CatNoise], S1=1/1000)

# Experiment 2
for n in [20, 60, 120]
        for model_type in ["fixed-q-v2-noise-fullycon", "interpretable-noise-fullycon", "interpretable-causal"]
            for replicate in 1:20
                println("Running for n = $n, neurons = $neurons, model = $model_type, replicate $replicate")
                data_split = BSON.load("constraints-paper/data/split_$(n)_$(replicate).bson")
                train = data_split[:train]
                test = data_split[:test]
                
                train_population = population[train]
                test_population = population[test]
                
                Threads.@threads for j in ["a", "b", "c", "d", "e"]
                    checkpoint_file = "constraints-paper/checkpoints/$(model_type)/$(covariates)/$(n)/checkpoint_set_$(replicate)$(j).bson"
                    if isfile(checkpoint_file)
                        println("[ALREADY EXISTS] replicate $j")
                        continue
                    end
                    println("replicate $j")
                    if model_type === "fixed-q-v2-noise-fullycon"
                        neurons_ = 32
                        ann = Flux.Chain(
                            Flux.Dense(2, neurons_, Flux.swish),
                            Flux.Dense(neurons_, 2),
                            AddFixedParameters([3, 4], 4, Flux.softplus),
                        )
                    elseif model_type === "interpretable-causal"
                        neurons_ = 32
                        nn_ffm = Flux.Chain(Flux.Dense(1, neurons_, Flux.swish), Flux.Dense(neurons_, 2, Flux.softplus))
                        nn_age = Flux.Chain(Flux.Dense(1, neurons_, Flux.swish), Flux.Dense(neurons_, 1, Flux.softplus))
                        ann = Flux.Chain(
                            Split(),
                            Flux.Parallel(Join(2, 1 => [1, 2], 2 => [1]), nn_ffm, nn_age), # note: not having age affect v1
                            Concatenate([3], 3, (x) -> Flux.sigmoid(x) * 0.4),
                            Concatenate([4], 4, (x) -> Flux.sigmoid(x) * 1.0),
                        )
                    elseif model_type === "interpretable-noise-fullycon"
                        neurons_ = 16
                        nn_ffm = Flux.Chain(Flux.Dense(1, neurons_, Flux.swish), Flux.Dense(neurons_, 2, Flux.softplus; bias=[0., 0.5]))
                        nn_age = Flux.Chain(Flux.Dense(1, neurons_, Flux.swish), Flux.Dense(neurons_, 2, Flux.softplus; bias=[0., 0.5]))
                        nn_noise = Flux.Chain(Flux.Dense(1, neurons_, Flux.swish), Flux.Dense(neurons_, 2, Flux.softplus; bias=[0., 0.5]))
                        nn_noise2 = Flux.Chain(Flux.Dense(1, neurons_, Flux.swish), Flux.Dense(neurons_, 2, Flux.softplus; bias=[0., 0.5]))
                        nn_catnoise = Flux.Chain(Flux.Dense(1, neurons_, Flux.swish), Flux.Dense(neurons_, 2, Flux.softplus; bias=[0., 0.5]))
                        ann = Flux.Chain(
                            Split(),
                            Flux.Parallel(Join(2, 1 => [1, 2], 2 => [1, 2], 3 => [1, 2], 4 => [1, 2], 5 => [1, 2]), nn_ffm, nn_age, nn_noise, nn_noise2, nn_catnoise),
                            Concatenate([3, 4], 4, Flux.softplus)
                        )
                    end
                    model = DCM(two_comp!, ann, 2)
                    optimizer = Flux.ADAM(1e-2)
                    fit!(model, train_population, optimizer; epochs=500)
                    BSON.bson(checkpoint_file, Dict(:model => model, :train => train, :test => test, :test_acc => sqrt(mse(model, test_population)), :optimizer => optimizer))
                end
            end
        end
    end
end

