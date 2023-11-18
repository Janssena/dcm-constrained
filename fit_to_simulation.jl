import Plots
import BSON
import Flux

using DeepCompartmentModels

include("src/lib/constraints.jl");

file = "constraints-paper/data/simulation-nhanes-chelle_additive_noise.csv"
# DATASET #1
population = load(file, [:Weight, :Height, :Age], S1=1/1000)
covariates = "wt-ht-age"
# DATASET #2
# population = load(file, [:FFM, :Age], S1=1/1000)
# covariates = "ffm-age"

# Experiment 1
for n in [20, 60, 120]
    for neurons in [8, 32, 128]
        for model_type in ["naive", "initialization", "normal-constraint", "fixed-q-v2"]
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

