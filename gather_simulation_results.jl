import Plots
import BSON
import Flux

include("src/lib/compartment_models.jl");
include("src/lib/dcm.jl");
include("src/lib/population.jl");
include("src/lib/dataset.jl");
include("src/lib/model.jl");
include("src/lib/objectives.jl");
include("src/lib/constraints.jl");
include("src/lib/ci.jl");

using Statistics
using DataFrames
using DeepCompartmentModels


# file = "constraints-paper/data/simulation-nhanes-chelle_additive.csv"
file = "constraints-paper/data/simulation-nhanes-chelle_additive_noise.csv"
df_true = DataFrame(CSV.File(file))
p_true = [Vector(group[1, [:CL, :V1, :Q, :V2]]) for group in groupby(df_true, :ID)]
population = load(file, [:Weight, :Height, :Age], S1=1/1000)

prob = ODEProblem(two_comp!, zeros(2), (-0.1, 72.))
t = collect(5/60:5/60:72.)
y = zeros(length(t), length(population))
for i in 1:length(population)
    prob_i = remake(prob, p=vcat(p_true[i], 0.))
    y[:, i] = solve(prob_i, saveat=t, save_idxs=1, tstops=callback=population.callbacks[i].condition.times, callback=population.callbacks[i]).u
end

# This calculates the rmse for the true curve vs the predicted curve within a precision of 5/60 minutes.
for covariates in ["wt-ht-age", "ffm-age"]
    if covariates == "ffm-age"
        population = load(file, [:FFM, :Age], S1=1/1000)
    end
    for neurons in [8, 32, 128]
        for model_type in ["naive", "initialization", "normal-constraint", "fixed-q-v2", "fixed-q-v2-noise-fullycon", "interpretable-noise-fullycon", "interpretable-causal"]
            print("Running for $(model_type)($neurons)... ")
            if contains(model_type, "noise") || contains(model_type, "interpretable")
                if neurons !== 32 continue end
                filename = "constraints-paper/data/results_ffm_age_noise_$(model_type)_neurons_$(neurons).csv" 
                population = load(file, [:FFM, :Age, :Noise, :Noise2, :CatNoise], S1=1/1000)
            else
                filename = "constraints-paper/data/results_$(replace(covariates, "-" => "_"))_$(model_type)_neurons_$(neurons).csv" 
            end
            if isfile(filename)
                println("EXISTS, MOVING ON ...")
                continue
            end
            df = DataFrame(n = vcat(repeat([120 60 20], 20, 1)...), replicate = repeat(1:20, 3), train_rmse_a_72 = 0., train_rmse_b_72 = 0., train_rmse_c_72 = 0., train_rmse_d_72 = 0., train_rmse_e_72 = 0., test_rmse_a_72 = 0., test_rmse_b_72 = 0., test_rmse_c_72 = 0., test_rmse_d_72 = 0., test_rmse_e_72 = 0., train_rmse_a_6 = 0., train_rmse_b_6 = 0., train_rmse_c_6 = 0., train_rmse_d_6 = 0., train_rmse_e_6 = 0., test_rmse_a_6 = 0., test_rmse_b_6 = 0., test_rmse_c_6 = 0., test_rmse_d_6 = 0., test_rmse_e_6 = 0., train_rmse_a_6_72 = 0., train_rmse_b_6_72 = 0., train_rmse_c_6_72 = 0., train_rmse_d_6_72 = 0., train_rmse_e_6_72 = 0., test_rmse_a_6_72 = 0., test_rmse_b_6_72 = 0., test_rmse_c_6_72 = 0., test_rmse_d_6_72 = 0., test_rmse_e_6_72 = 0.)
            for n in [20, 60, 120]
                for replicate in 1:20
                    Threads.@threads for j in ["a", "b", "c", "d", "e"] # This seems to lead to an error?
                        if contains(model_type, "noise") || contains(model_type, "interpretable")
                            checkpoint_file = "constraints-paper/checkpoints/$(model_type)/$(n)/checkpoint_set_$(replicate)$(j).bson"
                        else
                            checkpoint_file = "constraints-paper/checkpoints/$(model_type)/$(covariates)/$(neurons)/$(n)/checkpoint_set_$(replicate)$(j).bson"
                        end
                        tmp = BSON.parse(checkpoint_file) # the use of condition results in error when running @threads
                        checkpoint = BSON.raise_recursive(tmp, Main)
                        predictions = hcat(map(sol -> sol(t).u, predict(checkpoint[:model], population; tmax=72., interpolate=true))...)
                        squared_errors = (y - predictions).^2
                        idx = first((1:nrow(df))[(df.n .== n) .& (df.replicate .== replicate)])
                        df[idx, "train_rmse_$(j)_72"] = sqrt(mean(mean(squared_errors[:, checkpoint[:train]], dims=2)))
                        df[idx, "test_rmse_$(j)_72"] = sqrt(mean(mean(squared_errors[:, checkpoint[:test]], dims=2)))
                        df[idx, "train_rmse_$(j)_6"] = sqrt(mean(mean(squared_errors[1:length(5/60:5/60:6.), checkpoint[:train]], dims=2)))
                        df[idx, "test_rmse_$(j)_6"] = sqrt(mean(mean(squared_errors[1:length(5/60:5/60:6.), checkpoint[:test]], dims=2)))
                        df[idx, "train_rmse_$(j)_6_72"] = sqrt(mean(mean(squared_errors[length(5/60:5/60:6.):end, checkpoint[:train]], dims=2)))
                        df[idx, "test_rmse_$(j)_6_72"] = sqrt(mean(mean(squared_errors[length(5/60:5/60:6.):end, checkpoint[:test]], dims=2)))
                    end
                end
            end
            CSV.write(filename, df)
            println("DONE")
        end
    end
end
