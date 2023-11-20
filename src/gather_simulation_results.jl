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
                        # checkpoint_file = "constraints-paper/checkpoints/mse/additive-error/wt-ht-age/$(n)/$(neurons)/$(model_type)/checkpoint_set_$(replicate)$(j).bson"
                        # checkpoint_file = "constraints-paper/checkpoints/mse/additive-error/ffm-age/$(n)/$(neurons)/$(model_type)/checkpoint_set_$(replicate)$(j).bson"
                        # checkpoint_file = "/media/alexanderjanssen/T7 Touch/backup constraints paper checkpoints/additive-error/wt-ht-age/$(n)/$(neurons)/$(model_type)/checkpoint_set_$(replicate)$(j).bson"

                        # checkpoint_file = "/media/alexanderjanssen/T7 Touch/backup constraints paper checkpoints/additive-error/ffm-age/$(n)/$(neurons)/$(model_type)/checkpoint_set_$(replicate)$(j).bson"

                        # checkpoint_file = "constraints-paper/checkpoints/$(model_type)/$(n)/checkpoint_set_$(replicate)$(j).bson"
                        if contains(model_type, "noise") || contains(model_type, "interpretable")
                            checkpoint_file = "constraints-paper/checkpoints/$(model_type)/$(n)/checkpoint_set_$(replicate)$(j).bson"
                        else
                            checkpoint_file = "constraints-paper/checkpoints/$(model_type)/$(covariates)/$(neurons)/$(n)/checkpoint_set_$(replicate)$(j).bson"
                        end
                        # checkpoint = BSON.load(checkpoint_file)
                        try
                            tmp = BSON.parse(checkpoint_file) # the use of condition results in error when running @threads
                            delete!(tmp, :population)
                            checkpoint = BSON.raise_recursive(tmp, Main)
                        catch e
                            println("reading error for file $checkpoint_file:\n $e")
                            continue
                        end
                        tmp = BSON.parse(checkpoint_file) # the use of condition results in error when running @threads
                        delete!(tmp, :population)
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

# go through files and see what rows are still zero:
for neurons in [32]
    for model_type in ["normal-constraint"] #["naive", "initialization", "linear-constraint", "softsign-constraint", "normal-constraint", "fixed-q-v2", "normal-constraint-fixed-q-v2"]
        filename = "constraints-paper/data/results_wt_ht_age_$(model_type)_neurons_$(neurons).csv"
        # filename = "constraints-paper/data/results_ffm_age_noise_$(model_type)_neurons_$(neurons).csv" 
        df = DataFrame(CSV.File(filename))
        should_save = false
        println(file)
        for i in 1:nrow(df)
            if isapprox(sum(df[i, 3:end]), 0)
                # println("Filling in row: ", Vector(df[i, 3:end]))
                should_save = true
                n = df[i, :n]
                replicate = df[i, :replicate]
                Threads.@threads for j in ["a", "b", "c", "d", "e"] # This seems to lead to an error?
                    # checkpoint_file = "constraints-paper/checkpoints/mse/additive-error/wt-ht-age/$(n)/$(neurons)/$(model_type)/checkpoint_set_$(replicate)$(j).bson"
                    checkpoint_file = "/media/alexanderjanssen/T7 Touch/backup constraints paper checkpoints/additive-error/ffm-age-noise/$(n)/$(neurons)/$(model_type)/checkpoint_set_$(replicate)$(j).bson"
                    # checkpoint = BSON.load(checkpoint_file)
                    try
                        tmp = BSON.parse(checkpoint_file) # the use of condition results in error when running @threads
                        delete!(tmp, :population)
                        checkpoint = BSON.raise_recursive(tmp, Main)
                    catch 
                        println("reading error for file $checkpoint_file")
                        continue
                    end
                    tmp = BSON.parse(checkpoint_file) # the use of condition results in error when running @threads
                    delete!(tmp, :population)
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
                # println("After input: ", Vector(df[i, 3:end]))
            end
        end
        println("DONE")
        if should_save 
            CSV.write(filename, df) 
        end
    end
end


# df = DataFrame(n = repeat(vcat(repeat([120 60 20], 5, 1)...), 2), neurons = [fill(32, 15); fill(128, 15)])

# for fts in ["wt-ht-age", "ffm-age"]
#     covariates = fts === "wt-ht-age" ? [:Weight, :Height, :Age] : [:FFM, :Age]
#     population = load(file, covariates, S1=1/1000)
#     for neurons in [32, 128]
#         for n in [120, 60, 20]
#             rows = (df.n .== n) .& (df.neurons .== neurons)
#             for model_type in ["conditional-normal-constraint"] # ["tight-normal-constraint"] # ["linear-constraint", "softsign-constraint", "normal-constraint", "normal-constraint-fixed-q-v2"] # ["naive", "initialization", "linear-constraint", "normal-constraint", "softsign-constraint", "fixed-q-v2", "initialization-fixed-q-v2", "normal-constraint-fixed-q-v2"]
#                 results = zeros(5, 4)
#                 model_name = "mse-$(fts)-$(model_type)"
#                 for replicate in 1:5
#                     println("Running for n = $(n), neurons = $(neurons), model = $(model_name), replicate $(replicate)")
#                     train_loss = zeros(5)
#                     test_loss = zeros(5)
#                     for (i, j) in enumerate(["a", "b", "c", "d", "e"])
#                         path = "constraints-paper/checkpoints/mse/additive-error/$fts/$n/$neurons/$model_type/checkpoint_set_$(replicate)$(j).bson"
#                         checkpoint = BSON.load(path)

#                         model = checkpoint[:model]
#                         train = checkpoint[:train]
#                         test = checkpoint[:test]
#                         @assert all(isapprox.(checkpoint[:population].x, population.x)) "The saved population does not match the original population"

#                         train_loss[i] = mse(model, population[train])
#                         test_loss[i] = mse(model, population[test])

#                     end
#                     #append!(df, DataFrame(n = fill(n, 5), neurons = fill(neurons, 5), replicate = 1:5, model = model_name, train = mean(train_loss), train_std = std(train_loss), test = mean(test_loss), test_std = std(test_loss)))
#                     results[replicate, :] = [mean(train_loss), std(train_loss), mean(test_loss), std(test_loss)]
#                 end
#                 colnames = model_name .* ["-train", "-train-std", "-test", "-test-std"] 
#                 [name in names(df) ? nothing : (df[!, name] .= 0.) for name in colnames] # Add columns if not yet present.
#                 df[rows, colnames] = results 
#             end
#         end
#     end
# end

# CSV.write("constraints-paper/data/results_simulation_constraints.csv", df)



# Results from interpretable neural network:
file = "constraints-paper/data/simulation-nhanes-chelle_additive.csv"
population = load(file, [:FFM, :Age], S1=1/1000)
# file = "constraints-paper/data/simulation-nhanes-chelle_additive_noise.csv"
# population = load(file, [:FFM, :Age, :Noise, :CorFFM, :CorCL], S1=1/1000)

x_ = (0:0.001:1)'
med_ffm = normalize(55, tuple([x[1] for x in population.scale_x]...))
med_age = normalize(35, tuple([x[2] for x in population.scale_x]...))

as = zeros(2, 20)
# bs = zeros(2, 20)
bs = ones(2, 20) # For ageonv1 model
cs = zeros(2, 20)
ds = zeros(2, 20)
es = zeros(2, 20)
ffm_on_cl = zeros(length(x_), 20)
ffm_on_v1 = zeros(length(x_), 20)

age_on_cl = zeros(length(x_), 20)
age_on_v1 = zeros(length(x_), 20)

noise_on_cl = zeros(length(x_), 20)
noise_on_v1 = zeros(length(x_), 20)

corffm_on_cl = zeros(length(x_), 20)
corffm_on_v1 = zeros(length(x_), 20)

corcl_on_cl = zeros(length(x_), 20)
corcl_on_v1 = zeros(length(x_), 20)

for replicate in 1:20
    println("Running for replicate $(replicate)...")
    file = "constraints-paper/checkpoints/mse/additive-error/ffm-age/20/32/interpretable-ageonv1/checkpoint_set_$(replicate)a.bson"
    # file = "constraints-paper/checkpoints/mse/additive-error/ffm-age/$n/128/interpretable-noise/checkpoint_set_$(replicate)a.bson"
    checkpoint_tmp = BSON.parse(file)
    delete!(checkpoint_tmp, :population)
    checkpoint = BSON.raise_recursive(checkpoint_tmp, Main)

    ann = restructure(checkpoint[:model])

    ffm_model = ann.layers[2].layers[1]
    age_model = ann.layers[2].layers[2]
    # noise_model = ann.layers[2].layers[3]
    # corffm_model = ann.layers[2].layers[4]
    # corcl_model = ann.layers[2].layers[5]

    as[:, replicate] = ffm_model([med_ffm])
    ffm_on_cl[:, replicate] = ffm_model(x_)[1, :]
    ffm_on_v1[:, replicate] = ffm_model(x_)[2, :]
    
    bs[1, replicate] = first(age_model([med_age]))
    age_on_cl[:, replicate] = age_model(x_)[1, :]
    # age_on_v1[:, replicate] = age_model(x_)[2, :]

    # cs[:, replicate] = noise_model([0.5])
    # noise_on_cl[:, replicate] = noise_model(x_)[1, :]
    # noise_on_v1[:, replicate] = noise_model(x_)[2, :]
    
    # ds[:, replicate] = corffm_model([0.5])
    # corffm_on_cl[:, replicate] = corffm_model(x_)[1, :]
    # corffm_on_v1[:, replicate] = corffm_model(x_)[2, :]
    
    # es[:, replicate] = corcl_model([0.5])
    # corcl_on_cl[:, replicate] = corcl_model(x_)[1, :]
    # corcl_on_v1[:, replicate] = corcl_model(x_)[2, :]
end

# for n = 120
# idxs = vcat(2:4, 8, 9, 11:14, 16:19) # These are the good fits
# for n = 20:
# idxs = vcat(1:3, 8:9, 12, 14, 16:18, 20) # These are the good fits others have very low V1
# typical_values = mean((as .* bs .* cs .* ds .* es)[:, idxs], dims=2)[:, 1] # calculate based on the good fits
# factors = typical_values ./ (as .* bs .* cs .* ds .* es)

# For age on v1 model:
idxs = 1:20 # These are the good fits others have very low V1
typical_values = mean((as .* bs)[:, idxs], dims=2)[:, 1] # calculate based on the good fits
factors = typical_values ./ (as .* bs)

x_norm_ffm = [normalize_inv(x, tuple([s[1] for s in population.scale_x]...)) for x in x_]
x_norm_age = [normalize_inv(x, tuple([s[2] for s in population.scale_x]...)) for x in x_]

#################                      FFM                     #################

# FFM on CL:
plt_20_ffm_cl = Plots.plot(x_norm_ffm', (ffm_on_cl[:, idxs] ./ as[1, idxs]') .* factors[1, idxs]', legend=false, color=blue, linewidth=1.4)
Plots.plot!(x_norm_ffm', mean((ffm_on_cl[:, idxs] ./ as[1, idxs]') .* factors[1, idxs]', dims=2), color=:black, linewidth=2., xlabel="Fat free mass (kg)", ylabel="Fold change in typical CL", ylim=(0., 5))

# or:
# plt_120_cl = Plots.plot(x_norm_ffm', mean((ffm_on_cl[:, idxs] ./ as[1, idxs]') .* factors[1, idxs]', dims=2), ribbon=std((ffm_on_cl[:, idxs] ./ as[1, idxs]') .* factors[1, idxs]', dims=2), color=:black, fillcolor=blue, linewidth=1.4, fillalpha=0.4, xlabel="Fat free mass (kg)", ylabel="Factor change in typical CL", ylim=(0., 2.2), legend=false)

# FFM on V1
plt_20_ffm_v1 = Plots.plot(x_norm_ffm', (ffm_on_v1[:, idxs] ./ as[2, idxs]') .* factors[2, idxs]', legend=false, color=blue, linewidth=1.4)
Plots.plot!(x_norm_ffm', mean((ffm_on_v1[:, idxs] ./ as[2, idxs]') .* factors[2, idxs]', dims=2), color=:black, linewidth=2, xlabel="Fat free mass (kg)", ylabel="Factor change in typical V1")
# or:
# plt_120_v1 = Plots.plot(x_norm_ffm', mean((ffm_on_v1[:, idxs] ./ as[2, idxs]') .* factors[2, idxs]', dims=2), ribbon=std((ffm_on_v1[:, idxs] ./ as[2, idxs]') .* factors[2, idxs]', dims=2), color=:black, fillcolor=blue, linewidth=1.4, fillalpha=0.4, xlabel="Fat free mass (kg)", ylabel="Factor change in typical V1", legend=false)


#################                      AGE                     #################

# AGE on CL
plt_20_age_cl = Plots.plot(x_norm_age', (age_on_cl[:, idxs] ./ bs[1, idxs]') .* factors[1, idxs]', legend=false, color=orange, linewidth=1.4)
Plots.plot!(x_norm_age', mean((age_on_cl[:, idxs] ./ bs[1, idxs]') .* factors[1, idxs]', dims=2), color=:black, linewidth=2, xlabel="Age (years)", ylabel="Factor change in typical CL")

# or:
Plots.plot(x_norm_age', mean((age_on_cl[:, idxs] ./ bs[1, idxs]') .* factors[1, idxs]', dims=2), ribbon=std((age_on_cl[:, idxs] ./ bs[1, idxs]') .* factors[1, idxs]', dims=2), color=:black, fillcolor=orange, linewidth=1.4, fillalpha=0.4, xlabel="Age (years)", ylabel="Factor change in typical CL")

Plots.plot(plt_120_ffm_cl, plt_20_ffm_cl, plt_120_ffm_v1, plt_20_ffm_v1, plt_120_age_cl, plt_20_age_cl, layout=(3, 2), size=(600, 800))



plt_120 = Plots.plot(plta, pltb, pltc, layout=(1, 3), size=(1000, 300), bottommargin=5mm, leftmargin=5mm)

plt_20 = Plots.plot(plta, pltb, pltc, layout=(1, 3), size=(1000, 300), bottommargin=5mm, leftmargin=5mm)




# AGE on V1
Plots.plot(x_', (age_on_v1[:, idxs] ./ bs[2, idxs]') .* factors[2, idxs]', legend=false, color=:lightblue, linewidth=1.4)
Plots.plot!(x_', mean((age_on_v1[:, idxs] ./ bs[2, idxs]') .* factors[2, idxs]', dims=2), color=:black, linewidth=2, xlabel="Age (years)", ylabel="Factor change in typical V1")
# or
Plots.plot(x_', mean((age_on_v1[:, idxs] ./ bs[2, idxs]') .* factors[2, idxs]', dims=2), ribbon=std((age_on_v1[:, idxs] ./ bs[2, idxs]') .* factors[2, idxs]', dims=2), color=:black, fillcolor=:lightblue, linewidth=1.4, fillalpha=0.4, xlabel="Fat free mass (kg)", ylabel="Factor change in typical V1")


#################                     NOISE                    #################
label = ["Functions from replicates" fill(nothing, length(idxs) - 1)...]
green = "#a1d99b"
# NOISE on CL
pltc = Plots.plot(x_', (noise_on_cl[:, idxs] ./ cs[1, idxs]') .* factors[1, idxs]', color=green, linewidth=1.4, label=label, legend=false)
Plots.plot!(pltc, x_', mean((noise_on_cl[:, idxs] ./ cs[1, idxs]') .* factors[1, idxs]', dims=2), color=:black, linewidth=2, label="Mean", ylim=(0, 2))

# or:
Plots.plot(x_', mean((noise_on_cl[:, idxs] ./ cs[1, idxs]') .* factors[1, idxs]', dims=2), ribbon=std((noise_on_cl[:, idxs] ./ cs[1, idxs]') .* factors[1, idxs]', dims=2), color=:black, fillcolor=:lightblue, linewidth=1.4, fillalpha=0.4, xlabel="Noise", ylabel="Factor change in typical CL")

# AGE on V1
pltf = Plots.plot(x_', (noise_on_v1[:, idxs] ./ cs[2, idxs]') .* factors[2, idxs]', legend=false, linestyle=:dash, color=green, linewidth=1.4)
Plots.plot!(pltf, x_', mean((noise_on_v1[:, idxs] ./ cs[2, idxs]') .* factors[2, idxs]', dims=2), color=:black, linewidth=2, xlabel="Random noise", ylim=(0, 2))
# or
Plots.plot(x_', mean((noise_on_v1[:, idxs] ./ cs[2, idxs]') .* factors[2, idxs]', dims=2), ribbon=std((noise_on_v1[:, idxs] ./ cs[2, idxs]') .* factors[2, idxs]', dims=2), color=:black, fillcolor=:lightblue, linewidth=1.4, fillalpha=0.4, xlabel="Noise", ylabel="Factor change in typical V1")


#################                     CORFFM                   #################
blue = "#9ecae1"
# CORFFM on CL
plta = Plots.plot(x_', (corffm_on_cl[:, idxs] ./ ds[1, idxs]') .* factors[1, idxs]', color=blue, linewidth=1.4, label=label, legend=false)
Plots.plot!(plta, x_', mean((corffm_on_cl[:, idxs] ./ ds[1, idxs]') .* factors[1, idxs]', dims=2), color=:black, linewidth=2, ylabel="Fold change in typical CL", ylim=(0, 2), label="Mean")

# or:
Plots.plot(x_', mean((corffm_on_cl[:, idxs] ./ ds[1, idxs]') .* factors[1, idxs]', dims=2), ribbon=std((corffm_on_cl[:, idxs] ./ ds[1, idxs]') .* factors[1, idxs]', dims=2), color=:black, fillcolor=:lightblue, linewidth=1.4, fillalpha=0.4, xlabel="Noise", ylabel="Factor change in typical CL")

# CORFFM on V1
pltd = Plots.plot(x_', (corffm_on_v1[:, idxs] ./ ds[2, idxs]') .* factors[2, idxs]', color=blue, linestyle=:dash, linewidth=1.4, label=label, legend=false)
Plots.plot!(pltd, x_', mean((corffm_on_v1[:, idxs] ./ ds[2, idxs]') .* factors[2, idxs]', dims=2), color=:black, linewidth=2, xlabel="Covariate correlated with FFM", ylabel=L"\mathrm{Fold\ change\ in\ typical\ V}_1", ylim=(0, 2), label="Mean")
# or
Plots.plot(x_', mean((corffm_on_v1[:, idxs] ./ ds[2, idxs]') .* factors[2, idxs]', dims=2), ribbon=std((corffm_on_v1[:, idxs] ./ ds[2, idxs]') .* factors[2, idxs]', dims=2), color=:black, fillcolor=:lightblue, linewidth=1.4, fillalpha=0.4, xlabel="Noise", ylabel="Factor change in typical V1")


#################                     CORCL                    #################
orange = "#fdae6b"
# CORCL on CL
pltb = Plots.plot(x_', (corcl_on_cl[:, idxs] ./ es[1, idxs]') .* factors[1, idxs]', color=orange, linewidth=1.4, label=label, legend=false)
Plots.plot!(pltb, x_', mean((corcl_on_cl[:, idxs] ./ es[1, idxs]') .* factors[1, idxs]', dims=2), color=:black, linewidth=2, ylim = (0, 2), label="Mean")

# or:
Plots.plot(x_', mean((corffm_on_cl[:, idxs] ./ ds[1, idxs]') .* factors[1, idxs]', dims=2), ribbon=std((corffm_on_cl[:, idxs] ./ ds[1, idxs]') .* factors[1, idxs]', dims=2), color=:black, fillcolor=:lightblue, linewidth=1.4, fillalpha=0.4, xlabel="Noise", ylabel="Factor change in typical CL")

# CORCL on V1
plte = Plots.plot(x_', (corcl_on_v1[:, idxs] ./ ds[2, idxs]') .* factors[2, idxs]', legend=false, linestyle=:dash, color=orange, linewidth=1.4)
Plots.plot!(plte, x_', mean((corcl_on_v1[:, idxs] ./ ds[2, idxs]') .* factors[2, idxs]', dims=2), color=:black, linewidth=2, xlabel="Covariate correlated with CL", ylim=(0, 2))
# or
Plots.plot(x_', mean((corcl_on_v1[:, idxs] ./ ds[2, idxs]') .* factors[2, idxs]', dims=2), ribbon=std((corcl_on_v1[:, idxs] ./ ds[2, idxs]') .* factors[2, idxs]', dims=2), color=:black, fillcolor=:lightblue, linewidth=1.4, fillalpha=0.4, xlabel="Noise", ylabel="Factor change in typical V1")


Plots.plot(plta, pltb, pltc, pltd, plte, pltf, layout=(2, 3), size=(1000, 500), leftmargin=5mm, bottommargin=5mm)

# Plots.savefig("constraints-paper/plots/figure_4.pdf")
Plots.savefig("constraints-paper/plots/figure_5.pdf")
