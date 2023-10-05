datasets = ["stroke", "adult1994", "banknote2012", "creditfraud", "iris1988", "creditdefault2005", "coalmineseismicbumps"]#"stroke", "adult1994", "banknote2012", "creditfraud", "iris1988", "creditdefault2005", "coalmineseismicbumps", "yeast1996"
using DataFrames
using CSV
using DelimitedFiles
using Random
Random.seed!(1234);
using StatsBase
using Distances
using CategoricalArrays
using Gadfly, Cairo, Fontconfig, DataFrames, CSV
width = 6inch
height = 4inch
set_default_plot_size(width, height)
theme = Theme(major_label_font_size=16pt, minor_label_font_size=14pt, key_title_font_size=14pt, key_label_font_size=12pt, key_position=:none, colorkey_swatch_shape=:circle, key_swatch_size=12pt)
# Gadfly.push_theme(theme)

for dataset in datasets
	experiment_name = "ModelComparison"
    PATH = @__DIR__
    cd(PATH * "/Data/Tabular/$(dataset)")
	PATH = pwd()

	df = CSV.read("./Experiments/$(experiment_name)/df.csv", DataFrame, header=1)
	df2 = CSV.read("./Experiments/BNN_QBC/df.csv", DataFrame, header=1)
	df = filter(:AcquisitionFunction => ==("BNN_Random"), df)
	df = vcat(df, df2)
    fig1a = Gadfly.plot(df, x=:CumTrainedSize, y=:Accuracy, color=:AcquisitionFunction, Geom.point, Geom.line, yintercept=[0.5], Geom.hline(color=["red"], size=[1mm]), Guide.xlabel("Cumulative Training Size"), Coord.cartesian(xmin=df.AcquisitionSize[1]), Scale.y_continuous, theme) #, ymin=0.0, ymax=1.0

    fig1aa = Gadfly.plot(df, x=:CumTrainedSize, y=:Confidence, color=:AcquisitionFunction, Geom.point, Geom.line, yintercept=[0.5], Geom.hline(color=["red"], size=[1mm]), Guide.xlabel("Cumulative Training Size"), Coord.cartesian(xmin=df.AcquisitionSize[1]), Scale.y_continuous, theme)#, ymin=0.5, ymax=1.0

    fig1b = Gadfly.plot(df, x=:CumTrainedSize, y=:Elapsed, color=:AcquisitionFunction, Geom.point, Geom.line, Guide.ylabel("Training (seconds)"), Guide.xlabel("Cumulative Training Size"), Coord.cartesian(xmin=0), Scale.y_continuous, theme)

	# fig1d = plot(df, color=:AcquisitionFunction, Geom.blank,
        #   Guide.colorkey(title="Acquisition Function"))
	# plotgrid = gridstack([fig1a fig1aa; fig1b fig1d])

    fig1a |> PNG("./Experiments/$(experiment_name)/Accuracy_$(dataset)_$(experiment_name).png", dpi=600)
    fig1aa |> PNG("./Experiments/$(experiment_name)/Confidence_$(dataset)_$(experiment_name).png", dpi=600)
    fig1b |> PNG("./Experiments/$(experiment_name)/TrainingTime_$(dataset)_$(experiment_name).png", dpi=600)
    # plotgrid |> PNG("./Experiments/$(experiment_name)/ALLINFOPLOT.png", dpi=600)

	println(dataset)
end