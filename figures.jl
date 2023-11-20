import Plots
import BSON

using LaTeXStrings
using DeepCompartmentModels

blue_ = "#6baed6" # lighter: "#9ecae1", darker: "#358CBF", "#246083"
orange_ = "#fd8d3c" # lighter: "#fdae6b", darker: "#ea6402", "#b74e02"
green_ = "#74c476" # lighter: "#a1d99b"
purple_ = "#9a9ac8" # lighter: "#bcbddc", darker: "#6868ad", "#474782, "#2c2c51" <- 

################################################################################
##########                                                            ##########
##########                          Figure 1.                         ##########
##########                                                            ##########
################################################################################
# Schematic representation of hypothesis space:

##### how data was generated:
# uniform = kde(rand(Uniform(-15, 15), 100), bandwidth=1.4)
# normal = kde(rand(Normal(3, 2), 70), bandwidth=0.9)
# BSON.bson("plots/figure-1-uniform.bson", Dict(:x => uniform.x, :density => uniform.density))
uniform = BSON.load("plots/figure-1-uniform.bson")
normal = BSON.load("plots/figure-1-normal.bson")

Plots.plot(uniform[:x], uniform[:density] ./ maximum(uniform[:density]) .* 0.65, ribbon=(uniform[:density] ./ maximum(uniform[:density]) .* 0.65, 0.), linewidth=2, color=:grey, label=nothing, fillalpha=0.15)
Plots.plot!(normal[:x] .+ 0.35, normal[:density] ./ maximum(normal[:density]) .* 1.5, ribbon=(normal[:density] ./ maximum(normal[:density]) .* 1.5, 0.), color="#246083", fillcolor=blue_, linewidth=2, label=nothing, fillalpha=0.6)
Plots.plot!([-20, 22], [0, 0.], color=:black, linewidth=1.4, label=nothing)
Plots.plot!([4.68, 4.68], [0., 2], linestyle=:dash, color=:black, label=nothing)
Plots.scatter!([4.68], [0.], color=:black, label=nothing)
Plots.annotate!([4.68], [-0.1], Plots.text("True model", 10))
Plots.annotate!([1.], [-0.3], Plots.text(L"p(\theta\ \mid \mathcal{D})", 16))
Plots.annotate!([-17], [0.85], Plots.text("Naive model\nPoor inductive biases", 12, :left, "#5d5d5d"))
Plots.annotate!([6.], [1.65], Plots.text("Well-specified model\nPhysiological-based constraints", 12, :left, "#246083"))
Plots.plot!(xlim=(-20, 27), ylim=(-0.35, 2.2), guidefontsize=16, yaxis=false, yticks=false, xaxis=false, xticks=false)


################################################################################
##########                                                            ##########
##########                          Figure 2.                         ##########
##########                                                            ##########
################################################################################
# Made in Tikz


################################################################################
##########                                                            ##########
##########                          Figure 3.                         ##########
##########                                                            ##########
################################################################################
# Predicted concentration time curve in synthetic experiment.

file = "constraints-paper/data/simulation-nhanes-chelle_additive.csv"
population = load(file, [:Weight, :Height, :Age], S1=1/1000)
p = Matrix(DataFrame([group[1, [:CL, :V1, :Q, :V2, :TVCL, :TVV1]] for group in groupby(DataFrame(CSV.File(file)), :ID)]))

idx = 92

t = 0:0.1:75
prob = ODEProblem(two_comp!, zeros(2), (-0.1, 75.))
true_solution = zeros(length(t), 756)
for i in 1:length(population)
    prob_i = remake(prob, p=vcat(p[i, 1:4], 0.))
    true_solution[:, i] = solve(prob_i, saveat=t, save_idxs=1, tstops=callback=population.callbacks[i].condition.times, callback=population.callbacks[i]).u
end

res = zeros(20, length(t))
result = zeros(length(t), 756, 20)
model_type = "naive"
for i in 1:20 # Get the solution and plot the approximation to the solution space p(θ∣𝒟)
    checkpoint_file = "checkpoints/$(model_type)/wt-ht-age/32/20/checkpoint_set_$(i)a.bson"
    checkpoint_tmp = BSON.parse(checkpoint_file)
    delete!(checkpoint_tmp, :population) # Some error while loading the population
    checkpoint = BSON.raise_recursive(checkpoint_tmp, Main)
    res[i, :] = predict(checkpoint[:model], population[idx]; interpolate=true, tmax=75.)(t)
    result[:, :, i] = hcat(map(sol -> sol(t), predict(checkpoint[:model], population, interpolate=true, tmax=75.))...)
end

res2 = zeros(20, length(t))
result2 = zeros(length(t), 756, 20)
model_type = "normal-constraint"
for i in 1:20 # Get the solution and plot the approximation to the solution space p(θ∣𝒟)
    checkpoint_file = "checkpoints/$(model_type)/wt-ht-age/32/20/checkpoint_set_$(i)a.bson"
    checkpoint_tmp = BSON.parse(checkpoint_file)
    delete!(checkpoint_tmp, :population)
    checkpoint = BSON.raise_recursive(checkpoint_tmp, Main)
    res2[i, :] = predict(checkpoint[:model], population[idx]; interpolate=true, tmax=75.)(t)
    result2[:, :, i] = hcat(map(sol -> sol(t), predict(checkpoint[:model], population, interpolate=true, tmax=75.))...)
end

res3 = zeros(20, length(t))
result3 = zeros(length(t), 756, 20)
model_type = "fixed-q-v2"
for i in 1:20 # Get the solution and plot the approximation to the solution space p(θ∣𝒟)
    checkpoint_file = "checkpoints/$(model_type)/wt-ht-age/32/20/checkpoint_set_$(i)a.bson"
    checkpoint_tmp = BSON.parse(checkpoint_file)
    delete!(checkpoint_tmp, :population)
    checkpoint = BSON.raise_recursive(checkpoint_tmp, Main)
    res3[i, :] = predict(checkpoint[:model], population[idx]; interpolate=true, tmax=75.)(t)
    result3[:, :, i] = hcat(map(sol -> sol(t), predict(checkpoint[:model], population, interpolate=true, tmax=75.))...)
end

plta = Plots.plot(t, res', color=:darkgrey, alpha=0.5, linewidth=2)
Plots.plot!(t, median(res, dims=1)[1, :], color=:black, linewidth=2)
Plots.plot!(population[idx])
Plots.plot!(xlim=(-2, 72), ylim=(-0.2, 2.9), legend=false, xlabel="Time\n\n(a)", ylabel="FVIII level (IU/mL)")

pltb = Plots.plot(t, res2', color=blue, alpha=0.5, linewidth=2)
Plots.plot!(t, median(res2, dims=1)[1, :], color=:black, linewidth=2)
Plots.plot!(population[idx])
Plots.plot!(xlim=(-2, 72), ylim=(-0.2, 2.9), legend=false, xlabel="Time\n\n(b)")

pltc = Plots.plot(t, res3', color=green, alpha=0.5, linewidth=2, label=nothing)
Plots.plot!(t, median(res3, dims=1)[1, :], color=:black, linewidth=2, label="Median prediction")
Plots.plot!(population[idx], label="Observations")
Plots.plot!(xlim=(-2, 72), ylim=(-0.2, 2.9), xlabel="Time\n\n(c)", legend=:outerright)

Plots.plot(plta, pltb, pltc, layout=Plots.grid(1, 3, widths=[0.275, 0.275, 0.45]), size=(950, 275), bottommargin=10mm, leftmargin=5mm)

################################################################################
##########                                                            ##########
##########                          Figure 4.                         ##########
##########                                                            ##########
################################################################################
# Learned functions in synthetic experiment.

file = "constraints-paper/data/simulation-nhanes-chelle_additive.csv"
population_ = load(file, [:FFM, :Age], S1=1/1000)

x = collect(0:0.05:1)
ffm_n120 = zeros(length(x), 5 * 20)
res_n120 = zeros(length(x), 5 * 20)
cat_res_n120 = zeros(5, 5 * 20)
Threads.@threads for i in 1:20
    for (j, k) in enumerate(["a", "b", "c", "d", "e"])
        checkpoint_file = "constraints-paper/checkpoints/interpretable-noise-fullycon/120/checkpoint_set_$(i)$(k).bson"
        checkpoint_tmp = BSON.parse(checkpoint_file)
        checkpoint = BSON.raise_recursive(checkpoint_tmp, Main)
        ann = restructure(checkpoint[:model])
        ffm = ann.layers[2].layers[1]
        noise1 = ann.layers[2].layers[3]
        noise2 = ann.layers[2].layers[4]
        catnoise = ann.layers[2].layers[5]
        ffm_n120[:, ((j-1) * 20) + i] = (ffm(x') ./ ffm([0.6]))[1, :]
        res_n120[:, ((j-1) * 20) + i] = (noise2(x') ./ noise2([0.5]))[1, :]
        cat_res_n120[:, ((j-1) * 20) + i] = (catnoise(collect(0:0.25:1)') ./ catnoise([0.]))[1, :]
    end
end

ffm_n20 = zeros(length(x), 5 * 20)
res_n20 = zeros(length(x), 5 * 20)
cat_res_n20 = zeros(5, 5 * 20)
Threads.@threads for i in 1:20
    for (j, k) in enumerate(["a", "b", "c", "d", "e"])
        checkpoint_file = "constraints-paper/checkpoints/interpretable-noise-fullycon/20/checkpoint_set_$(i)$(k).bson"
        checkpoint_tmp = BSON.parse(checkpoint_file)
        checkpoint = BSON.raise_recursive(checkpoint_tmp, Main)
        ann = restructure(checkpoint[:model])
        ffm = ann.layers[2].layers[1]
        noise1 = ann.layers[2].layers[3]
        noise2 = ann.layers[2].layers[4]
        catnoise = ann.layers[2].layers[5]
        ffm_n20[:, ((j-1) * 20) + i] = (ffm(x') ./ ffm([0.6]))[1, :]
        res_n20[:, ((j-1) * 20) + i] = (noise2(x') ./ noise2([0.5]))[1, :]
        cat_res_n20[:, ((j-1) * 20) + i] = (catnoise(collect(0:0.25:1)') ./ catnoise([0.]))[1, :]
    end
end

scale_ffm = map(s -> s[1], population_.scale_x)

plt0 = Plots.plot(normalize_inv(x, scale_ffm), ffm_n120, color=blue, ylim=(0, 1.9), alpha=0.33, label=nothing)
Plots.plot!(normalize_inv(x, scale_ffm), mean(ffm_n120, dims=2)[:, 1], color=:black, linewidth=3, label="Average")
Plots.plot!(xlabel="Fat-free mass (kg)", ylabel="Fold change in clearance", legend=:bottomright)

plt00 = Plots.plot(normalize_inv(x, scale_ffm), ffm_n20, color=orange, legend=false, ylim=(0, 1.9), alpha=0.33)
Plots.plot!(normalize_inv(x, scale_ffm), mean(ffm_n20, dims=2)[:, 1], color=:black, linewidth=3)
Plots.plot!(xlabel="Fat free mass (kg)", ylabel="Fold change in clearance")

plt1 = Plots.plot(x, res_n120, color=blue, legend=false, ylim=(0.75, 1.25), alpha=0.33)
Plots.plot!(x, mean(res_n120, dims=2)[:, 1], color=:black, linewidth=3)
Plots.plot!(xlabel="Continuous noise")

plt2 = Plots.plot(x, res_n20, color=orange, legend=false, ylim=(0.75, 1.25), alpha=0.33)
Plots.plot!(x, mean(res_n20, dims=2)[:, 1], color=:black, linewidth=3)
Plots.plot!(xlabel="Continuous noise")

plt3 = Plots.scatter(rand(Normal(0, 0.1), (52, 5))' .+ collect(1:5), cat_res_n120, markersize=3, color=blue, markerstrokecolor=blue, label=nothing)
Plots.scatter!(1:5, mean(cat_res_n120, dims=2)[:, 1], yerr=std(cat_res_n120, dims=2)[:, 1], color=:black, markersize=4, label=nothing)
Plots.plot!(xlabel="Categorical noise", ylim=(0.67, 1.37), legend=:bottomleft)

plt4 = Plots.scatter(rand(Normal(0, 0.1), (52, 5))' .+ collect(1:5), cat_res_n20, markersize=3, markerstrokecolor=orange, color=orange, legend=false)
Plots.scatter!(1:5, mean(cat_res_n20, dims=2)[:, 1], yerr=std(cat_res_n20, dims=2)[:, 1], color=:black, markersize=4, legend=false)
Plots.plot!(xlabel="Categorical noise", ylim=(0.67, 1.37))

bottom = Plots.plot(plt0, plt1, plt3, plt00, plt2, plt4, layout=(2, 3))

Plots.plot(bottom, size=(950, 600), guidefontsize=10, leftmargin=5mm, bottommargin=5mm, markerstrokewidth=0.5, framestyle=:box)



################################################################################
##########                                                            ##########
##########                   Supplementary figure 1.                   ##########
##########                                                            ##########
################################################################################
# Made in Tikz


################################################################################
##########                                                            ##########
##########                   Supplementary figure 2.                   ##########
##########                                                            ##########
################################################################################
# Predicted concentration time curve in real-world experiment.

file = "data/opticlot.csv" # This is private data.
df = DataFrame(CSV.File(file))
filter!(row -> row.ID !== 59, df)

df[df.Dose .!== 0., :Rate] = df[df.Dose .!== 0., :Dose] .* 60
df[df.Dose .!== 0., :Duration] .= 1/60
population = load(df, [:Weight, :Height, :Age, :BGO], S1=1/1000)
population.x .= collect(normalize_inv(population.x', population.scale_x)')

types = ["naive", "bound", "global", "inn"]
t = 0:(5/60):48.
solution = zeros(length(types), 10, length(t))
plt = Plots.plot()
k = 1
for (j, type) in enumerate(types)
    for i in 1:10
        filename = "$(type)_fold_$(i)_10_folds.bson"
        ckpt = BSON.load(joinpath("checkpoints/comparison/final_manuscript", filename))
        _, re = Flux.destructure(ckpt[:ann])
        ann = re(ckpt[:weights])
        model = DCM(two_comp!, ann, 2)

        solution[j, i, :] = predict(model, population[k]; interpolate = true)(t)
        Plots.plot!(plt, sol)
    end
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
Plots.plot!(plt4, population[k], markersize=6, label="Observations", legend=false)

Plots.plot(plt1, plt2, plt3, plt4, framestyle=:box, layout=(1, 4), size=(1400, 300), xlabel="Time", ylim=(-0.1, 2.2), yticks=(0:0.5:2, string.(0:50:200)), bottommargin=8mm, leftmargin=8mm)


################################################################################
##########                                                            ##########
##########                   Supplementary figure 3.                   ##########
##########                                                            ##########
################################################################################
# Learned functions in real-world experiment.

i = 1
ckp = BSON.parse("checkpoints/comparison/causal_fold_$(i).bson")
BSON.raise_recursive(ckp, Main)

x = 0:0.01:1
wtht_on_cl = zeros(length(x), 10)
age_on_cl = zeros(length(x), 10)
bgo_on_cl = zeros(10)
for i in 1:10
    filename = "inn_fold_$(i)_10_folds.bson"
    ckpt = BSON.load(joinpath("checkpoints/comparison/final_manuscript", filename))
    _, re = Flux.destructure(ckpt[:ann])
    model = re(ckpt[:weights])
    age_on_cl[:, i] = Float64.(model.layers[3].layers[2](collect(x)')') ./ model.layers[3].layers[2]([0.2])
    bgo_on_cl[i] = first(model.layers[3].layers[3]([1])) / first(model.layers[3].layers[3]([0]))
end

filename = "inn_fold_1_10_folds.bson"
ckpt = BSON.load(joinpath("checkpoints/comparison/final_manuscript", filename))
_, re = Flux.destructure(ckpt[:ann])
model = re(ckpt[:weights])

plta = Plots.plot()
Plots.surface!(0:0.05:1, 0:0.05:1, (x1, x2) -> model.layers[3].layers[1]([x1, x2])[1] / model.layers[3].layers[1]([0.5, 0.8])[1], color=:RdBu_11, clim=(0, 2))
Plots.wireframe!((0:30:150) ./ 150, [40; 50:25:200; 210] ./ 210, fillalpha=0, (x1, x2) -> model.layers[3].layers[1]([x1, x2])[1] / model.layers[3].layers[1]([0.5, 0.8])[1])
Plots.plot!(xticks=(0:0.2:1, string.(Integer.((0:0.2:1) .* 150))), framestyle=:box, grid=true, yticks=(round.(50:25:210) ./ 210, 50:25:210), ylim=(40/210, 1.), zlabel="Fold change in\ntypical clearance", xlabel="Weight (kg)", ylabel="Height (cm)")

pltb = Plots.plot()
Plots.surface!(0:0.05:1, 0:0.05:1, (x1, x2) -> model.layers[3].layers[1]([x1, x2])[2] / model.layers[3].layers[1]([0.5, 0.8])[2], color=:RdBu_11, clim=(0, 2))
Plots.wireframe!((0:30:150) ./ 150, [40; 50:25:200; 210] ./ 210, fillalpha=0, (x1, x2) -> model.layers[3].layers[1]([x1, x2])[2] / model.layers[3].layers[1]([0.5, 0.8])[2])
Plots.plot!(xticks=(0:0.2:1, string.(Integer.((0:0.2:1) .* 150))), framestyle=:box, grid=true, yticks=(round.(50:25:210) ./ 210, 50:25:210), ylim=(40/210, 1.), zlabel="Fold change in typical\nvolume of distribution", xlabel="Weight (kg)", ylabel="Height (cm)")

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

# Note: The wireframe plot in julia adds a white background to the wireframe, we 
# removed the vectors manually in an image editor.

