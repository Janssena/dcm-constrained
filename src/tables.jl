import CSV

using Statistics
using DataFrames

function result(x_) 
    x = Matrix(x_) .* 100
    med = median(x) 
    sd = std(x)
    div = sum(x .> med * 1.5)
    return "$(round(med, sigdigits=3)) +- $(round(sd, sigdigits=2)) ($div)"
end

# Table 1:
for neurons in [8, 32, 128]
    for covariates in ["wt_ht_age", "ffm_age"]
        for model_type in ["naive", "initialization", "normal-constraint", "fixed-q-v2"]
            df = DataFrame(CSV.File("constraints-paper/data/results_$(covariates)_$(model_type)_neurons_$(neurons).csv"))
            println("($covariates, $model_type, $neurons neurons):\t\t| n = 20 $(result(df[41:end, 8:12])) | n = 60 $(result(df[21:40, 8:12])) | n = 120 $(result(df[1:20, 8:12])) | ")
        end
    end
    println()
end

# Table 2:
for model_type in ["fixed-q-v2", "interpretable-fullycon", "interpretable-ageonv1"]
    file = "constraints-paper/data/results_ffm_age_noise_$(model_type)_neurons_32.csv"
    df = DataFrame(CSV.File(file))
    println("($model_type):\t\t| n = 20 $(result(df[41:end, 8:12])) | n = 60 $(result(df[21:40, 8:12])) | n = 120 $(result(df[1:20, 8:12])) | ")
end
