import Optimisers
import Random
import Zygote
import Plots
import BSON
import CSV

include("src/lib/compartment_models.jl");
include("src/lib/dcm.jl");
include("src/lib/population.jl");
include("src/lib/dataset.jl");
include("src/lib/model.jl");
include("src/lib/objectives.jl");
include("src/lib/constraints.jl");
include("src/lib/ci.jl");
include("constraints-paper/src/node.jl");

using Lux
using DataFrames
using Statistics

for num_neurons in [8, 32]
    for hidden_dim in [1, 2]
        for latent_dim in [2, 6]
            Threads.@threads for fold in 1:10
                ckpt_file = "constraints-paper/checkpoints/comparison/final_manuscript/node/node_fold_$(fold)_10_folds_neurons_$(num_neurons)_latentdim_$(latent_dim)_hiddendim_$(hidden_dim).bson"
                if isfile(ckpt_file) continue end
                # get population
                file = "constraints-paper/data/opticlot_train_with_val_set_$(fold)_10_folds.csv"
                df = DataFrame(CSV.File(file))
                df[df.Dose .!== 0., :Rate] = df[df.Dose .!== 0., :Dose] .* 60
                df[df.Dose .!== 0., :Duration] .= 1/60
                population = load(df, [:Weight, :Height, :Age, :BGO], S1=1/1000) # Note: S1 is technically another hyperparameter
                population.x .= collect(normalize_inv(population.x', population.scale_x)')

                file_val = "constraints-paper/data/opticlot_val_$(fold)_10_folds.csv"
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

                # Roughly the models as reported in original paper:
                encoder = Lux.Chain(
                    Normalize(Float32[150, 210, 100, 2]),
                    Lux.Dense(4, num_neurons, Lux.relu),
                    Lux.Dense(num_neurons, latent_dim * 2) # to μ and σ
                )

                if hidden_dim == 1
                    neuralode = Lux.Chain( # Replaced the selu for tanh to improve stability
                        Lux.Dense(latent_dim, num_neurons, Lux.tanh),
                        Lux.Dense(num_neurons, latent_dim)
                    )
                else
                    neuralode = Lux.Chain( # Replaced the selu for tanh to improve stability
                        Lux.Dense(latent_dim, num_neurons, Lux.tanh),
                        Lux.Dense(num_neurons, num_neurons, Lux.tanh),
                        Lux.Dense(num_neurons, latent_dim)
                    )
                end

                decoder = Lux.Chain(
                    Lux.Dense(latent_dim, num_neurons, Lux.selu),
                    Lux.Dense(num_neurons, 1, Lux.softplus)
                )

                ps1, st1 = Lux.setup(Random.default_rng(), encoder)
                ps2, st2 = Lux.setup(Random.default_rng(), neuralode)
                ps3, st3 = Lux.setup(Random.default_rng(), decoder)
                parameters = ComponentArray(enc = ps1, node = ps2, dec = ps3, I = 0.f0)
                states = (enc = st1, node = st2, dec = st3,)
                
                opt = Optimisers.ADAM(1e-3)
                opt_state = Optimisers.setup(opt, parameters)
                opt_params = copy(parameters)

                best_val_loss = objective(encoder, neuralode, decoder, population_val, parameters, states)
                for epoch in 1:4_000
                    loss, back = Zygote.pullback(p -> objective(encoder, neuralode, decoder, population, p, states), parameters)
                    if epoch == 1 || epoch % 250 == 0
                        println("(Fold = $fold) Epoch $epoch: loss = $loss")
                    end
                    ∇ = first(back(1))
                    opt_state, parameters = Optimisers.update(opt_state, parameters, ∇)
                    parameters.I = 0.f0

                    new_val_loss = objective(encoder, neuralode, decoder, population_val, parameters, states)
                    if new_val_loss < best_val_loss
                        # save parameters
                        opt_params = copy(parameters)
                        # reset best val loss  
                        best_val_loss = new_val_loss
                        # println(" SAVED")
                    else
                        # println()
                    end
                end
                k = length(vcat(test_population.y...)) # number of measurements

                # estimate accuracy using 100 Monte Carlo evaluations of the objective:
                val_sse = mean([objective(encoder, neuralode, decoder, population_val, opt_params, states) for _ in 1:100])
                # SSE / k = mse
                test_mse = [objective(encoder, neuralode, decoder, test_population, opt_params, states) / k for _ in 1:100]
                test_rmse = sqrt(mean(test_mse))
                
                ckpt = Dict(:p_encoder => NamedTuple(opt_params.enc), :p_neuralode => NamedTuple(opt_params.dec), :p_decoder => NamedTuple(opt_params.node), :st_encoder => states.enc, :st_neuralode => states.dec, :st_decoder => states.node, :encoder => encoder, :neuralode => neuralode, :decoder => decoder, :val_sse => val_sse, :test_rmse => test_rmse)
                BSON.bson(ckpt_file, ckpt)
            end
        end
    end
end


folder = "constraints-paper/checkpoints/comparison/final_manuscript/node"
files = readdir(folder)
vals = zeros(10)
test = zeros(10)
for (i, file) in enumerate(filter(contains("neurons_32_latentdim_2_hiddendim_1"), files))
    ckpt = BSON.load(joinpath(folder, file));
    vals[i] = ckpt[:val_sse]
    test[i] = ckpt[:test_rmse]
end
println("$(mean(vals)) ($(median(vals))) +- $(std(vals))")


# mean (median) +- std
# 8 neurons, latent 2, hidden 1
500.57283678431884 (1.5637119529804095) +- 1576.742258042253 "(one replicate got into bad local optima)"
# 8 neurons, latent 2, hidden 2
7.395767513698097 (6.933498965162758) +- 4.836732429411328
# 8 neurons, latent 6, hidden 1
7.616917329678159 (6.933498965162758) +- 4.499827460266068
# 8 neurons, latent 6, hidden 2
3.576334318126734 (1.6852916093972694) +- 5.117063699980006

# 32 neurons, latent 2, hidden 1
1.1353877558700018 (1.0929523115533604) +- 0.4950705273562451 "<<<<<"
# 32 neurons, latent 2, hidden 2
1.6460264891496792 (0.9461179991290954) +- 2.132444582585326
# 32 neurons, latent 6, hidden 1
3.833330156710763 (4.085063386075815) +- 1.6176334151213614
# 32 neurons, latent 6, hidden 2
"""Hangs during optimization"""



# Make plot:
file = "constraints-paper/data/opticlot.csv"
df = DataFrame(CSV.File(file))
filter!(row -> row.ID !== 59, df)

df[df.Dose .!== 0., :Rate] = df[df.Dose .!== 0., :Dose] .* 60
df[df.Dose .!== 0., :Duration] .= 1/60
population = load(df, [:Weight, :Height, :Age, :BGO], S1=1/1000)
population.x .= collect(normalize_inv(population.x', population.scale_x)')

folder = "constraints-paper/checkpoints/comparison/final_manuscript/node"
files = readdir(folder)
t = 0:(5/60):48.
solution = zeros(10, length(t))
# plt = Plots.plot()
k = 1
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
    plt = Plots.plot()
    res = zeros(length(t))
    for _ in 1:100
        z = softplus.(θ[d+1:end]) .* randn(d) + θ[1:d]
        # z′ = predict_node(ckpt[:neuralode], individual, z, p, st)(t)
        z′ = predict_node(ckpt[:neuralode], individual, z, p, st)(0:0.1:24*6)
        ŷ, _ = ckpt[:decoder](hcat(z′.u...), p.dec, st.dec)
        # res += ŷ[1, :]
        Plots.plot!(plt, 0:0.1:24*6, ŷ[1, :], color=Plots.palette(:default)[7], label=nothing)
    end
    Plots.plot!(plt, xlim=(-0.5, 72), ylim=(-0.1, 2))

    
    solution[i, :] = res ./ 100
end

Plots.plot(t, solution', color=:dodgerblue, label=nothing)
Plots.plot!(t, mean(solution, dims=1)[1, :], linewidth=3, color=:black, label=nothing)

