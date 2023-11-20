import Random
import CSV

include("src/lib/dcm.jl")
include("src/lib/population.jl")
include("src/lib/dataset.jl")

using Statistics
using DataFrames
using Plots.Measures
# Prepare data folds:
file = "data/opticlot.csv" # This data is private.
df = DataFrame(CSV.File(file))

filter!(row -> row.ID !== 59, df) # Remove individual with incorrect historical data

train_, test_ = create_cv(unique(df.ID), 10);

for i in 1:10
    train = train_[i]
    test = test_[i]
    df_group = groupby(df, :ID) 
    CSV.write("constraints-paper/data/opticlot_train_$(i)_10_folds.csv", DataFrame(df_group[train]))
    CSV.write("constraints-paper/data/opticlot_test_$(i)_10_folds.csv", DataFrame(df_group[test]))
end

# Create val sets for ML models
for i in 1:10
    df_train = DataFrame(CSV.File("constraints-paper/data/opticlot_train_$(i)_10_folds.csv"))
    idxs = unique(df_train.ID)
    Random.shuffle!(idxs)
    val_cutoff = Integer(round(0.2 * length(idxs)))
    val = idxs[1:val_cutoff] # 11 samples used for validation
    new_train = idxs[val_cutoff+1:end]
    df_new_train = filter(row -> row.ID ∈ new_train, df_train)
    df_val = filter(row -> row.ID ∈ val, df_train)
    CSV.write("constraints-paper/data/opticlot_train_with_val_set_$(i)_10_folds.csv", df_new_train)
    CSV.write("constraints-paper/data/opticlot_val_$(i)_10_folds.csv", df_val)
end
import Zygote
import Plots
import BSON
import Flux

include("src/lib/constraints.jl");

using StatsPlots
using DeepCompartmentModels

# Fit models
for model_type in ["naive", "bound", "global", "inn"]
    Threads.@threads for i in 1:10
        file = "constraints-paper/data/opticlot_train_with_val_set_$(i)_10_folds.csv"
        df = DataFrame(CSV.File(file))
        df[df.Dose .!== 0., :Rate] = df[df.Dose .!== 0., :Dose] .* 60
        df[df.Dose .!== 0., :Duration] .= 1/60
        population = load(df, [:Weight, :Height, :Age, :BGO], S1=1/1000)
        population.x .= collect(normalize_inv(population.x', population.scale_x)')

        file_val = "constraints-paper/data/opticlot_val_$(i)_10_folds.csv"
        df_val = DataFrame(CSV.File(file_val))
        df_val[df_val.Dose .!== 0., :Rate] = df_val[df_val.Dose .!== 0., :Dose] .* 60
        df_val[df_val.Dose .!== 0., :Duration] .= 1/60
        population_val = load(df_val, [:Weight, :Height, :Age, :BGO], S1=1/1000)
        population_val.x .= collect(normalize_inv(population_val.x', population_val.scale_x)')
        
        df_test = DataFrame(CSV.File(replace(file_val, "_val" => "_test")))
        df_test[df_test.Dose .!== 0., :Rate] = df_test[df_test.Dose .!== 0., :Dose] .* 60
        df_test[df_test.Dose .!== 0., :Duration] .= 1/60
        test_population = load(df_test, [:Weight, :Height, :Age, :BGO], S1=1/1000)
        test_population.x .= collect(normalize_inv(test_population.x', test_population.scale_x)')

        if model_type == "naive"
            ann = Flux.Chain(
                x -> x ./ [150.f0, 210.f0, 100.f0, 1.f0],
                Flux.Dense(4, 32, Flux.swish),
                Flux.Dense(32, 4, Flux.softplus),
            )
        elseif model_type == "bound"
            ann = Flux.Chain(
                x -> x ./ [150.f0, 210.f0, 100.f0, 1.f0],
                Flux.Dense(4, 32, Flux.swish),
                Flux.Dense(32, 4),
                NormalConstraint([0.0, 0.3, 0.05, 0.], [0.5, 7., 0.5, 2.])
            )
        elseif model_type == "global"
            ann = Flux.Chain(
                x -> x ./ [150.f0, 210.f0, 100.f0, 1.f0],
                Flux.Dense(4, 32, Flux.swish),
                Flux.Dense(32, 2),
                AddFixedParameters([3, 4], 4, Flux.softplus)
            )
        elseif model_type == "inn"
            neurons_ = 16
            nn_wtht = Flux.Chain(Flux.Dense(2, neurons_, Flux.swish), Flux.Parallel(vcat, Flux.Dense(neurons_, 1, Flux.softplus), Flux.Dense(neurons_, 1, Flux.softplus; bias=[0.5])))
            nn_bgo = Flux.Chain(Flux.Dense(1, 1, Flux.softplus; bias=[0.5]))
            nn_age = Flux.Chain(Flux.Dense(1, neurons_, Flux.swish), Flux.Dense(neurons_, 1, Flux.softplus; bias=[0.5]))
            
            ann = Flux.Chain(
                x -> x ./ [150.f0, 210.f0, 100.f0, 1.f0],
                Split([1,2], [3], [4]),
                Flux.Parallel(Join(2, 1 => [1, 2], 2 => [1], 3 => [1]), nn_wtht, nn_age, nn_bgo), # note: not having age affect v1
                Concatenate([3, 4], 4, Flux.softplus),
            )
        end

        model = DCM(two_comp!, ann, 2)
        opt = Flux.ADAM(3e-3)

        p_opt = copy(model.weights)
        best_val_loss = mse(model, population_val)
        for epoch in 1:4000
            idxs = unique(rand(1:length(population), 35))
            ∇ = first(Zygote.gradient(p -> mse(model, population[idxs], p), model.weights))
            if epoch > 200 && mse(model, population_val) < best_val_loss
                p_opt = copy(model.weights)
                best_val_loss = mse(model, population_val)
                println("[SET $(i)] Epoch $epoch SAVED. VAL RMSE = $(sqrt(best_val_loss))")
            end
            Flux.update!(opt, model.weights, ∇)
        end
        ckpt = Dict(:weights => p_opt, :ann => ann, :val_rmse => sqrt(best_val_loss), :test_rmse => sqrt(mse(model, test_population, p_opt)))
        filename = "$(model_type)_fold_$(i)_10_folds.bson"
        BSON.bson(joinpath("constraints-paper/checkpoints/comparison/final_manuscript", filename), ckpt)
    end
end

res = zeros(10)
for i in 1:10
    # filename = "naive_fold_$(i)_10_folds.bson"
    filename = "bound_fold_$(i)_10_folds.bson"
    # filename = "global_fold_$(i)_10_folds.bson"
    # filename = "inn_fold_$(i)_10_folds.bson"
    ckpt = BSON.load(joinpath("constraints-paper/checkpoints/comparison/final_manuscript", filename))
    
    res[i] = ckpt[:test_rmse]
end
# NAIVE
14.802308027028953 +- 2.808190836311825
# BOUNDARY
14.14665497578333 +- 2.3733602669998994
# GLOBAL
13.86686970323408 +- 1.849121654552172
# MULTI-BRANCH
13.012984956101553 +- 2.077315367356427
