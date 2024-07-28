datasets = ["iris1988"]#"stroke", "adult1994", "banknote2012", "creditfraud", "creditdefault2005", "coalmineseismicbumps", "iris1988", "yeast1996"
minimum_training_sizes = [30] #40, 60, 40, 40, 80, 100, 30, 296
acquisition_sizes = round.(Int, minimum_training_sizes ./ 10)

y_ticks = collect(0:0.1:1.0)

experiments = ["IncrementalLearning", "IncrementalLearningXGB", "IncrementalLearningDNN"]
sub_experiments = ["Random", "Random", "Random"]

using DelimitedFiles
using Random
Random.seed!(1234);
using StatsBase
using Gadfly, Cairo, Fontconfig, DataFrames, CSV
width = 6inch
height = 6inch
set_default_plot_size(width, height)
theme = Theme(major_label_font_size=16pt, minor_label_font_size=14pt, key_title_font_size=14pt, key_label_font_size=12pt, key_position=:top, colorkey_swatch_shape=:circle, key_swatch_size=12pt)
Gadfly.push_theme(theme)

for (dataset, acquisition_size, max_training_size) in zip(datasets, acquisition_sizes, minimum_training_sizes)
    x_ticks = collect(acquisition_size:acquisition_size:max_training_size)

    println(dataset)
    PATH = @__DIR__
    cd(PATH * "/Data/Tabular/$(dataset)")

    ###
    ### Data
    ###

    PATH = pwd()
    df_acc = DataFrame()
    df_f1 = DataFrame()

    for (experiment, sub_experiment) in zip(experiments, sub_experiments)
        df_acc_ = CSV.read("./Experiments/$(experiment)/mean_std_acc$(sub_experiment).csv", DataFrame, header=1)
        df_f1_ = CSV.read("./Experiments/$(experiment)/mean_std_f1$(sub_experiment).csv", DataFrame, header=1)

        insertcols!(df_acc_, :Experiment => experiment)
        insertcols!(df_f1_, :Experiment => experiment)

        df_acc = vcat(df_acc, df_acc_)
        df_f1 = vcat(df_f1, df_f1_)
    end


    fig1a = Gadfly.plot(df_acc, x=:CumulativeTrainedSize, y=:WeightedAccuracy_mean, color=:Experiment, ymin=df_acc.Accuracy_mean - df_acc.Accuracy_std, ymax=df_acc.Accuracy_mean + df_acc.Accuracy_std, Geom.point, Geom.line, Geom.ribbon, Guide.ylabel("Accuracy"), Guide.xlabel("Cumulative Training Size"), Guide.yticks(ticks=y_ticks), Guide.xticks(ticks=x_ticks), Coord.cartesian(xmin=df_acc.CumulativeTrainedSize[1], ymin=0.0, ymax=1.0))
    fig1aa = Gadfly.plot(df_f1, x=:CumulativeTrainedSize, y=:WeightedF1_mean, color=:Experiment, ymin=df_f1.F1_mean - df_f1.F1_std, ymax=df_f1.F1_mean + df_f1.F1_std, Geom.point, Geom.line, Geom.ribbon, Guide.ylabel("F1"), Guide.xlabel("Cumulative Training Size"), Guide.yticks(ticks=y_ticks), Guide.xticks(ticks=x_ticks), Coord.cartesian(xmin=df_f1.CumulativeTrainedSize[1], ymin=0.0, ymax=1.0))

    fig1a |> PDF("./Experiments/Model_comparison_Accuracy_$(dataset).pdf", dpi=300)
    fig1aa |> PDF("./Experiments/Model_comparison_F1_$(dataset).pdf", dpi=300)

end