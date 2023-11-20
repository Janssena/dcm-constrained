import Random
import CSV

using Statistics
using DataFrames
using DeepCompartmentModels

file = "data/opticlot.csv"
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