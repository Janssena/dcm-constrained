import Random
import CSV

include("src/lib/dcm.jl")
include("src/lib/population.jl")
include("src/lib/dataset.jl")

using Statistics
using DataFrames
using Plots.Measures

blue_ = "#6baed6" # lighter: "#9ecae1"
orange_ = "#fd8d3c" # lighter: "#fdae6b"
green_ = "#74c476" # lighter: "#a1d99b"
purple_ = "#9a9ac8" # lighter: "#bcbddc"

font = "Computer Modern"
# font = "cmr10" # for pyplot
Plots.default(fontfamily=font, tick_direction=:out, axiscolor=Plots.Colors.RGBA(0,0,0,0.5), titlefontsize=12, framestyle=:axes, grid=false)

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


# Append columns
# orig_file = "../OPTI-CLOT/data/complete_dataframe_with_surgduration_nacl.csv"
# df_orig = DataFrame(CSV.File(orig_file))
# df_orig_ = DataFrame([group[1, [:APTT, :PT, :Albumine]] for group in groupby(df_orig, :SubjectID)])
# for col in [:APTT, :PT, :Albumine]
#     df_orig_[!, col] = convert.(Union{Missing, Float64}, df_orig_[:, col])
#     df_orig_[ismissing.(df_orig_[:, col]), col] .= mean(skipmissing(df_orig_[:, col]))
# end
# ids = filter(!=(59), unique(DataFrame(CSV.File("constraints-paper/data/opticlot.csv")).ID))
# df_orig_[!, :ID] = ids

# for i in 1:5
#     for sub in ["train", "test"]
#         df__ = DataFrame(CSV.File("constraints-paper/data/opticlot_$(sub)_$(i).csv"))
#         CSV.write("constraints-paper/data/opticlot_$(sub)_$(i)_ext.csv", leftjoin(df__, df_orig_,on=:ID))
#     end
# end

import Zygote
import Plots
import BSON
import Flux

include("src/lib/constraints.jl");
include("src/lib/ci.jl");

using StatsPlots
using DeepCompartmentModels

# Fit models
for num_neurons in [8, 32, 128]
    for model_type in ["naive", "bound", "global", "inn"]
        Threads.@threads for i in 1:10
            ckpt_file = "constraints-paper/checkpoints/comparison/final_manuscript/$(model_type)/$(model_type)_fold_$(i)_10_folds_neurons_$num_neurons.bson"
            file = "constraints-paper/data/opticlot_train_with_val_set_$(i)_10_folds.csv"
            df = DataFrame(CSV.File(file))
            df[df.Dose .!== 0., :Rate] = df[df.Dose .!== 0., :Dose] .* 60
            df[df.Dose .!== 0., :Duration] .= 1/60
            population = load(df, [:Weight, :Height, :Age, :BGO], S1=1/1000)
            # population = load(df, [:Weight, :Age, :BGO], S1=1/1000)
            population.x .= collect(normalize_inv(population.x', population.scale_x)')
            # population = load(df, [:Weight, :Height, :Age, :BGO, :APTT, :PT, :Albumine], S1=1/1000)

            file_val = "constraints-paper/data/opticlot_val_$(i)_10_folds.csv"
            df_val = DataFrame(CSV.File(file_val))
            df_val[df_val.Dose .!== 0., :Rate] = df_val[df_val.Dose .!== 0., :Dose] .* 60
            df_val[df_val.Dose .!== 0., :Duration] .= 1/60
            population_val = load(df_val, [:Weight, :Height, :Age, :BGO], S1=1/1000)
            # population_val = load(df_val, [:Weight, :Age, :BGO], S1=1/1000)
            population_val.x .= collect(normalize_inv(population_val.x', population_val.scale_x)')
            
            df_test = DataFrame(CSV.File(replace(file_val, "_val" => "_test")))
            df_test[df_test.Dose .!== 0., :Rate] = df_test[df_test.Dose .!== 0., :Dose] .* 60
            df_test[df_test.Dose .!== 0., :Duration] .= 1/60
            test_population = load(df_test, [:Weight, :Height, :Age, :BGO], S1=1/1000)
            # test_population = load(df_test, [:Weight, :Age, :BGO], S1=1/1000)
            # test_population = load(df_test, [:Weight, :Height, :Age, :BGO, :APTT, :PT, :Albumine], S1=1/1000)
            test_population.x .= collect(normalize_inv(test_population.x', test_population.scale_x)')

            if model_type == "naive"
                ann = Flux.Chain(
                    x -> x ./ [150.f0, 210.f0, 100.f0, 1.f0],
                    Flux.Dense(4, num_neurons, Flux.swish),
                    Flux.Dense(num_neurons, 4, Flux.softplus),
                )
            elseif model_type == "bound"
                ann = Flux.Chain(
                    x -> x ./ [150.f0, 210.f0, 100.f0, 1.f0],
                    Flux.Dense(4, num_neurons, Flux.swish),
                    Flux.Dense(num_neurons, 4),
                    NormalConstraint([0.0, 0.3, 0.05, 0.], [0.5, 7., 0.5, 2.])
                )
            elseif model_type == "global"
                ann = Flux.Chain(
                    x -> x ./ [150.f0, 210.f0, 100.f0, 1.f0],
                    Flux.Dense(4, num_neurons, Flux.swish),
                    Flux.Dense(num_neurons, 2),
                    AddFixedParameters([3, 4], 4, Flux.softplus)
                )
            elseif model_type == "inn"
                num_neur = Integer(num_neurons / 2)
                nn_wtht = Flux.Chain(
                    Flux.Dense(2, num_neur, Flux.swish), 
                    Flux.Parallel(vcat, 
                    Flux.Dense(num_neur, 1, Flux.softplus), 
                    Flux.Dense(num_neur, 1, Flux.softplus; bias=[0.5])
                    )
                )
                nn_bgo = Flux.Chain(
                    Flux.Dense(1, 1, Flux.softplus; bias=false)
                )
                nn_age = Flux.Chain(
                    Flux.Dense(1, num_neur, Flux.swish), 
                    Flux.Dense(num_neur, 1, Flux.softplus; bias=[0.5])
                )
                
                ann = Flux.Chain(
                    x -> x ./ [150.f0, 210.f0, 100.f0, 1.f0],
                    Split([1,2], [3], [4]),
                    Flux.Parallel(Join(2, 1 => [1, 2], 2 => [1], 3 => [1]), nn_wtht, nn_age, nn_bgo), # note: not having age affect v1
                    # x -> x ./ [150.f0, 100.f0, 1.f0],
                    # Split(),
                    # Flux.Parallel(Join(2, 1 => [1, 2], 2 => [1], 3 => [1]), nn_wt, nn_age, nn_bgo), # note: not having age affect v1
                    Concatenate([3, 4], 4, Flux.softplus),
                    # Concatenate([3], 3, (x) -> Flux.sigmoid(x) * 0.4),
                    # Concatenate([4], 4, (x) -> Flux.sigmoid(x) * 1.0),
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
            # cb(m, test) = (e, l) -> println("Epoch $(e): RMSE = $(sqrt(l)), test = $(sqrt(mse(m, test)))")
            # fit!(model, population, opt; epochs=500, callback=cb(model, test_population))
            # [13.65, 14.76, 14.72, 15.88, 14.05] seconds

            # display(Plots.plot(predict(model, test_population[1]; interpolate=true)))
            ckpt = Dict(:weights => p_opt, :ann => ann, :val_rmse => sqrt(best_val_loss), :test_rmse => sqrt(mse(model, test_population, p_opt)))
            # filename = "$(model_type)_fold_$(i)_10_folds_neurons_$num_neurons.bson"
            BSON.bson(ckpt_file, ckpt)
            # BSON.bson("constraints-paper/checkpoints/comparison/naive_fold_$i.bson", Dict(:model => model, :weights => model.weights, :train_rmse => sqrt(mse(model, population)), :test_rmse => sqrt(mse(model, test_population))))
            # BSON.bson("constraints-paper/checkpoints/comparison/causal_fold_$(i)_new_with_val_set.bson", Dict(:model => model.re(p_opt), :train_rmse => sqrt(mse(model, population, p_opt)), :test_rmse => sqrt(mse(model, test_population, p_opt))))
        end
    end
end

map(i -> BSON.parse("constraints-paper/checkpoints/comparison/naive_fold_$(i)_new_with_val_set.bson")[:test_rmse], 1:5)
# 0.1480111092417415 +- 0.026993270674340205

map(i -> BSON.load("constraints-paper/checkpoints/comparison/global_fold_$(i)_new_with_val_set.bson")[:test_rmse], 1:5)
# 0.13830399432507737 +- 0.025735911430812525

map(i -> BSON.load("constraints-paper/checkpoints/comparison/causal_fold_$(i)_new_with_val_set.bson")[:test_rmse], 1:5)
# 0.13484563496280383 +- 0.023175639602231728

res = zeros(10)
for i in 1:10
    # filename = "naive_fold_$(i)_10_folds.bson"
    filename = "bound_fold_$(i)_10_folds.bson"
    # filename = "global_fold_$(i)_10_folds.bson"
    # filename = "inn_fold_$(i)_10_folds.bson"
    # filename = "inn_weight_fold_$(i)_10_folds.bson"
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
# MULTI-BRANCH weight
14.277785404553335 +- 1.5931502547594008


# Acc of Bjorkman 
# 
# with BGO:
# 15.06 +- 1.9455

# Acc of McEneny-King:
#
# with BGO:
# 13.9565 +- 2.1676


# Visualize learned functions
i = 1
ckp = BSON.parse("constraints-paper/checkpoints/comparison/causal_fold_$(i).bson")
BSON.raise_recursive(ckp, Main)




# Learned effects:
x = 0:0.01:1
wtht_on_cl = zeros(length(x), 10)
age_on_cl = zeros(length(x), 10)
bgo_on_cl = zeros(10)
for i in 1:10
    filename = "inn_fold_$(i)_10_folds.bson"
    ckpt = BSON.load(joinpath("constraints-paper/checkpoints/comparison/final_manuscript", filename))
    _, re = Flux.destructure(ckpt[:ann])
    model = re(ckpt[:weights])
    # wtht_on_cl[:, :, i] = model.layers[3].layers[1]
    age_on_cl[:, i] = Float64.(model.layers[3].layers[2](collect(x)')') ./ model.layers[3].layers[2]([0.2])
    bgo_on_cl[i] = first(model.layers[3].layers[3]([1])) / first(model.layers[3].layers[3]([0]))
end

filename = "inn_fold_1_10_folds.bson"
ckpt = BSON.load(joinpath("constraints-paper/checkpoints/comparison/final_manuscript", filename))
_, re = Flux.destructure(ckpt[:ann])
model = re(ckpt[:weights])

plta = Plots.plot()
Plots.surface!(0:0.05:1, 0:0.05:1, (x1, x2) -> model.layers[3].layers[1]([x1, x2])[1] / model.layers[3].layers[1]([0.5, 0.8])[1], color=:RdBu_11, clim=(0, 2))
# Plots.wireframe!(0:0.1:1, 0:0.1:1, fillalpha=0, (x1, x2) -> model.layers[3].layers[1]([x1, x2])[1] / model.layers[3].layers[1]([0.5, 0.8])[1])
Plots.wireframe!((0:30:150) ./ 150, [40; 50:25:200; 210] ./ 210, fillalpha=0, (x1, x2) -> model.layers[3].layers[1]([x1, x2])[1] / model.layers[3].layers[1]([0.5, 0.8])[1])
Plots.plot!(xticks=(0:0.2:1, string.(Integer.((0:0.2:1) .* 150))), framestyle=:box, grid=true, yticks=(round.(50:25:210) ./ 210, 50:25:210), ylim=(40/210, 1.), zlabel="Fold change in\ntypical clearance", xlabel="Weight (kg)", ylabel="Height (cm)")
Plots.savefig("constraints-paper/plots/wtht_cl_learned_functions_realworld.svg")

pltb = Plots.plot()
Plots.surface!(0:0.05:1, 0:0.05:1, (x1, x2) -> model.layers[3].layers[1]([x1, x2])[2] / model.layers[3].layers[1]([0.5, 0.8])[2], color=:RdBu_11, clim=(0, 2))
# Plots.wireframe!(0:0.1:1, 0:0.1:1, fillalpha=0, (x1, x2) -> model.layers[3].layers[1]([x1, x2])[2] / model.layers[3].layers[1]([0.5, 0.8])[2])
Plots.wireframe!((0:30:150) ./ 150, [40; 50:25:200; 210] ./ 210, fillalpha=0, (x1, x2) -> model.layers[3].layers[1]([x1, x2])[2] / model.layers[3].layers[1]([0.5, 0.8])[2])
Plots.plot!(xticks=(0:0.2:1, string.(Integer.((0:0.2:1) .* 150))), framestyle=:box, grid=true, yticks=(round.(50:25:210) ./ 210, 50:25:210), ylim=(40/210, 1.), zlabel="Fold change in typical\nvolume of distribution", xlabel="Weight (kg)", ylabel="Height (cm)")
Plots.savefig("constraints-paper/plots/wtht_v1_learned_functions_realworld.svg")

pltc = Plots.plot()
Plots.hline!([1], color=:lightgrey, linestyle=:dash, label=nothing)
Plots.plot!(x, age_on_cl, linewidth=2, color=orange_, alpha=0.4, label=nothing)
Plots.plot!(x, mean(age_on_cl, dims=2), linewidth=2, color=:black, label="Mean effect")
Plots.plot!(xticks = (0:0.2:1, string.(0:20:100)), xlabel="Age (in years)", ylabel="Fold change in clearance", xlim=(-0.05, 1.05), ylim=(-0.05, 1.5))

pltd = Plots.plot()
Plots.vline!([1], color=:lightgrey, linestyle=:dash)
Plots.scatter!([mean(bgo_on_cl)], [1.], xerr=[1.96 * std(bgo_on_cl)], markersize=5, color=green_)
Plots.scatter!([1], [0.], color=green_, markersize=5)
Plots.plot!(yticks=(0:1, ["non-O", "O"]), ylabel="Blood group", legend=false, ylim=(-0.5, 1.5), xlim=(0.7, 1.7), xlabel="Fold change in clearance")

bottom = Plots.plot(pltc, pltd, framestyle=:box, size=(700, 300), bottommargin=5mm, leftmargin=5mm)
Plots.savefig("constraints-paper/plots/bottom_learned_functions_realworld.svg")


# Look at predicted concentration time curve:
file = "constraints-paper/data/opticlot.csv"
df = DataFrame(CSV.File(file))
filter!(row -> row.ID !== 59, df)

df[df.Dose .!== 0., :Rate] = df[df.Dose .!== 0., :Dose] .* 60
df[df.Dose .!== 0., :Duration] .= 1/60
population = load(df, [:Weight, :Height, :Age, :BGO], S1=1/1000)
population.x .= collect(normalize_inv(population.x', population.scale_x)')

types = ["naive", "bound", "global", "inn"]
t = 0:(5/60):48.
solution = zeros(length(types) + 1, 10, length(t))
k = 1
for (j, type) in enumerate(types)
    for i in 1:10
        filename = "$(type)_fold_$(i)_10_folds.bson"
        ckpt = BSON.load(joinpath("constraints-paper/checkpoints/comparison/final_manuscript", filename))
        _, re = Flux.destructure(ckpt[:ann])
        ann = re(ckpt[:weights])
        model = DCM(two_comp!, ann, 2)

        solution[j, i, :] = predict(model, population[k]; interpolate = true)(t)
    end
end

# N-ODE:
longer_t = 0:(5/60):72.
longer_solution = zeros(10, length(longer_t))
folder = "constraints-paper/checkpoints/comparison/final_manuscript/node"
files = readdir(folder)
for (i, file) in enumerate(filter(contains("neurons_32_latentdim_2_hiddendim_1"), files))
    println("Running for file $i")
    ckpt = BSON.load(joinpath(folder, file))
    individual = population[k] 
    # Decoder and neuralode parameters are swapped in ckpt
    p = ComponentVector(enc = ckpt[:p_encoder], node = ckpt[:p_decoder], dec = ckpt[:p_neuralode], I = 0.f0)
    st = (enc = ckpt[:st_encoder], node = ckpt[:st_neuralode], dec = ckpt[:st_decoder])
     
    θ, _ = ckpt[:encoder](individual.x, p.enc, st.enc)
    d = Integer(size(θ, 1) / 2)
    
    # 100 Monte Carlo samples:
    res = zeros(length(t))
    res2 = zeros(length(longer_t))
    for _ in 1:100
        z = softplus.(θ[d+1:end]) .* randn(d) + θ[1:d]
        z′ = predict_node(ckpt[:neuralode], individual, z, p, st)(t)
        longer_z′ = predict_node(ckpt[:neuralode], individual, z, p, st)(longer_t)
        ŷ, _ = ckpt[:decoder](hcat(z′.u...), p.dec, st.dec)
        longer_ŷ, _ = ckpt[:decoder](hcat(longer_z′.u...), p.dec, st.dec)
        res += ŷ[1, :]
        res2 += longer_ŷ[1, :]
    end
    
    solution[end, i, :] = res ./ 100
    longer_solution[i, :] = res2 ./ 100
end

plt1 = Plots.plot()
Plots.plot!(plt1, t, solution[1, :, :]', color=:darkgrey, alpha=0.4, linewidth=2, label=["Single replicate" fill(nothing, 1, 9)...])
Plots.plot!(plt1, t, median(solution[1, :, :], dims=1)', color=:black, linewidth=2, label="Median prediction")
Plots.plot!(population[k], markersize=6, label="Observations", ylabel="FVIII level (IU/dL)")

plt2 = Plots.plot()
Plots.plot!(plt2, t, solution[2, :, :]', color=blue_, alpha=0.4, linewidth=2)
Plots.plot!(plt2, t, median(solution[2, :, :], dims=1)', color=:black, linewidth=2)
Plots.plot!(plt2, population[k], markersize=6, label="Observations", legend=false)

plt3 = Plots.plot()
Plots.plot!(plt3, t, solution[3, :, :]', color=green_, alpha=0.4, linewidth=2)
Plots.plot!(plt3, t, median(solution[3, :, :], dims=1)', color=:black, linewidth=2)
Plots.plot!(plt3, population[k], markersize=6, label="Observations", legend=false)

plt4 = Plots.plot()
Plots.plot!(plt4, t, solution[4, :, :]', color=purple_, alpha=0.6, linewidth=2)
Plots.plot!(plt4, t, median(solution[4, :, :], dims=1)', color=:black, linewidth=2)
Plots.plot!(plt4, population[k], markersize=6, label="Observations", ylabel="FVIII level (IU/dL)", legend=false)

plt5 = Plots.plot()
Plots.plot!(plt5, t, solution[5, :, :]', color=Plots.palette(:default)[7], alpha=0.6, linewidth=2)
Plots.plot!(plt5, t, median(solution[5, :, :], dims=1)', color=:black, linewidth=2)
Plots.plot!(plt5, population[k], markersize=6, label="Observations", legend=false)


Plots.plot(plt1, plt2, plt3, plt4, plt5, framestyle=:box, layout=(2, 3), size=(800, 500), xlabel="Time", ylim=(-0.1, 2.2), yticks=(0:0.5:2, string.(0:50:200)), bottommargin=8mm, leftmargin=8mm)
Plots.savefig("constraints-paper/plots/figure_supplement_realworld_curves_new.svg")



Plots.plot(longer_t, longer_solution', color=Plots.palette(:default)[7], alpha=0.6, linewidth=2, label=["Single replicate" fill(nothing, 1, 9)...])
Plots.plot!(longer_t, median(longer_solution, dims=1)', color=:black, linewidth=2, label="Median prediction")
Plots.plot!(population[k], markersize=6, label="Observations", xlabel="Time", ylabel="FVIII level (IU/dL)", framestyle=:box, size=(400, 300))
Plots.savefig("constraints-paper/plots/figure_supplement_node_extrapolation.svg")