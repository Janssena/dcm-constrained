import DifferentialEquations: ODEProblem, remake, solve
import Plots
import CSV

using DataFrames
using Distributions
using DeepCompartmentModels

function ci(p, cb, Ω; α::Real=0.05, sensitivity::Float64=15/60) # a point every 15 minutes
    η_ci = confidence_ellipsoid(Ω, α)
    range = collect(-0.1:sensitivity:72.)
    sols = zeros(length(range), size(η_ci, 2))

    for i in 1:size(sols, 2)
        sols[:, i] = solve(remake(prob, p=vcat(p .* exp.([η_ci[:, i]; zeros(2)]), 0.)), saveat=range, tstops=cb.condition.times, save_idxs=1, callback=cb).u
    end

    typ = solve(remake(prob, p=vcat(p, 0.)), saveat=range, tstops=cb.condition.times, save_idxs=1, callback=cb).u
    ribbon = (typ - minimum(sols, dims=2)[:, 1], maximum(sols, dims=2)[:, 1] - typ)

    return (x = range, y = typ, ribbon = ribbon)
end


# data from the US National Health and Nutrition Examination Survey (NHANES):
df = DataFrame(CSV.File("constraints-paper/simulation/NHANES_miss.csv"))

selection = (.!ismissing.(df.AgeMonths)) .& (.!ismissing.(df.Weight)) .& (.!ismissing.(df.Height)) .& (df.Gender .== "male")
df_group = groupby(df[selection, :], :ID)

idxs = unique(rand(1:length(df_group), 1000)) # take a random subset of individuals!

df_ = unique(DataFrame(df_group[idxs]))
df_[!, :Age] = df_.AgeMonths ./ 12.
df_ = df_[:, [:ID, :Gender, :Age, :Weight, :Height, :BMI]]

# prevalence of O⁺ and O⁻ blood types is 35% + 13%.
df_[!, :BGO] .= rand.(Bernoulli(0.35 + 0.13))

# From "Influence of blood group, von Willebrand factor levels, and age on factor VIII levels in non-severe haemophilia"
# We derive two distributions from which we sample VWFAg
# vwf_bgo = LogNormal(0.07, 0.25)
# vwf_nonbgo = LogNormal(0.3, 0.35)
# df_[!, :VWFAg] .= df_.BGO .* rand.(vwf_bgo) + .!df_.BGO .* rand.(vwf_nonbgo)


# Choose PK model. Might be nice if we include more complex relationships.

# ours: not suitable for children likely?
# Devine's equation:
# ibw = @. 50. + 2.3 * ((df_.Height - 152.4) / 2.54)
# ibw = @. 39. + 2.27 * ((df_.Height - 152.4) / 2.54) # Traubs equation
# # Traub's equation for children below 152.4 cm:
# ibw[df_.Height .< 152.4] = @. (df_.Height[df_.Height .< 152.4]^2 * 1.65) / 1000
# # The above leads to quite a substantial discontinuity which we do not want.
# Ω = [0.0296]
# σ² = [0.027, 0.156] # Combined error.
# tvcl = @. 93.3 * (0.5 * log.(exp.(0.132 * (ibw - 40.6)) + 1) + 1) * 0.789^(df_.Age .>= 45) * 1.26^df_.BGO * ((1 - 0.247) + (0.247 / (1 + exp.(-1 * (-20.3 * (df_.VWFAg - 0.912))))))
# cl = tvcl * exp.(rand.(Normal(0., sqrt(first(Ω)))))
# v1 = @. 1010 * (1 + (ibw^3 / (0.234 * 1000000)))
# q = 155
# v2 = 722

# WAPPS FFM model
# Calculated using al-sallami's function for men:
ffm = @. (0.88 + ((1-0.88) / (1 + (df_.Age / 13.4)^-12.7))) * ((9270 * df_.Weight) / (6680 + (216 * df_.BMI)))

Ω = [0.306^2 0.01; 0.01 0.191^2] # is CV, but as std of η according to paper
σ = [0.05] # Additive error for OSA
η = rand(MultivariateNormal(zeros(2), Ω), nrow(df_))

tvcl = @. 0.168 * (ffm / 54.6)^0.941 * (1 + -0.178 * max.(0, df_.Age - 19) / 19)
cl = tvcl .* exp.(η[1, :])
tvv1 = @. 2.78 * (ffm / 54.6)^0.985
v1 = tvv1 .* exp.(η[2, :])
q = 0.126
v2 = 0.446

# Plots.scatter(df_.Weight, ibw)
# Plots.scatter!(df_.Weight, ffm)

include("src/lib/compartment_models.jl");
include("src/lib/dcm.jl");
include("src/lib/population.jl");
include("src/lib/dataset.jl");

prob = ODEProblem(two_comp!, zeros(2), (-0.1, 72.))

pred = zeros(nrow(df_), 3)
ipred = zeros(nrow(df_), 3)

doses = round.((50. .* df_.Weight) ./ 250.) .* 250.

df_final = DataFrame()

for i in 1:nrow(df_)
    row = df_[i, :]
    t = max.([4, 24, 48] + rand.([Normal(0., 2.), Normal(0., 5.), Normal(0., 5.)]), [15/60, 0., 0.]) # Measurements are no sooner than 15 minutes after dose.
    prob_typ = remake(prob, p=[tvcl[i], tvv1[i], q, v2, 0.])
    prob_indv = remake(prob, p=[cl[i], v1[i], q, v2, 0.])
    𝐈 = [0. doses[i] doses[i] .* 60. 1/60; 1/60 0. 0. 0.]
    callback = first(DeepCompartmentModels._generate_dosing_callbacks([𝐈]; S1=1/1000))

    pred = solve(prob_typ, saveat=t, tstops=callback.condition.times, save_idxs=1, callback=callback).u
    ipred = solve(prob_indv, saveat=t, tstops=callback.condition.times, save_idxs=1, callback=callback).u

    dv = max.(ipred .+ rand.(Normal(0., first(σ))) + ipred .* rand.(Normal(0., 0.16)), 0.)

    new_row = repeat(DataFrame(row), length(t) + 1)
    # new_row[!, :IBW] .= ibw[i]
    new_row[!, :FFM] .= ffm[i]
    new_row[!, :Time] = [0.; t]
    new_row[!, :Dose] = [doses[i]; zeros(length(t))]
    new_row[!, :Rate] = new_row.Dose .* 60.
    new_row[!, :Duration] = [1/60; zeros(length(t))]
    new_row[!, :PRED] = [0.; pred]
    new_row[!, :IPRED] = [0.; ipred]
    new_row[!, :DV] = [0.; dv]
    new_row[!, :MDV] = [1; zeros(length(t))]
    new_row[!, :TVCL] .= tvcl[i]
    new_row[!, :TVV1] .= tvv1[i]
    new_row[!, :CL] .= cl[i]
    new_row[!, :V1] .= v1[i]
    new_row[!, :Q] .= q
    new_row[!, :V2] .= v2
    new_row[!, :ETA1] .= η[1, i]
    new_row[!, :ETA2] .= η[2, i]

    append!(df_final, new_row)
end

CSV.write("constraints-paper/data/simulation-nhanes-chelle_additive.csv", df_final)

df_final[!, :Noise] .= 0.
df_final[!, :Noise2] .= 0.
df_final[!, :CatNoise] .= 0.
for group in groupby(df_final, :ID)
    group.Noise .= rand()
    group.Noise2 .= rand()
    group.CatNoise .= rand(1:5) ./ 5
end

CSV.write("constraints-paper/data/simulation-nhanes-chelle_additive_noise.csv", df_final)
