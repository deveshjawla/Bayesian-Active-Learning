
credit_fraud_df = CSV.read("Data/creditcard.csv", DataFrame, header=true)
stroke_df = CSV.read("Data/healthcare-dataset-stroke-data.csv", DataFrame, header=true)

stroke_df_no_bmi = stroke_df[stroke_df.bmi.=="N/A", :]

stroke_df_no_bmi = select(stroke_df_no_bmi, Not(:bmi))

println(eltype.(eachcol(stroke_df_11)))

stroke_df = stroke_df[stroke_df.bmi.!="N/A", :]
println(eltype.(eachcol(stroke_df)))
println(names(stroke_df))

stroke_df[!, :bmi] = parse.(Float32, stroke_df[!, :bmi])


stroke_df_no_smoking_status = stroke_df[stroke_df.smoking_status.=="Unknown", :]
stroke_df_no_smoking_status = select(stroke_df_no_smoking_status, Not(:smoking_status))

stroke_df = stroke_df[stroke_df.smoking_status.!="Unknown", :]

stroke_df = stroke_df_no_smoking_status
stroke_df = stroke_df[stroke_df.gender.!="Other", :]
stroke_df[stroke_df.gender.=="Male", :gender] .= "1"
stroke_df[stroke_df.gender.=="Female", :gender] .= "2"
stroke_df[!, :gender] = parse.(Int32, stroke_df[!, :gender])

stroke_df[stroke_df.ever_married.=="Yes", :ever_married] .= "1"
stroke_df[stroke_df.ever_married.=="No", :ever_married] .= "0"
stroke_df[!, :ever_married] = parse.(Int32, stroke_df[!, :ever_married])

stroke_df[stroke_df.work_type.=="Private", :work_type] .= "1"
stroke_df[stroke_df.work_type.=="Self-employed", :work_type] .= "2"
stroke_df[stroke_df.work_type.=="Govt_job", :work_type] .= "3"
stroke_df[stroke_df.work_type.=="children", :work_type] .= "4"
stroke_df[stroke_df.work_type.=="Never_worked", :work_type] .= "5"
stroke_df[!, :work_type] = parse.(Int32, stroke_df[!, :work_type])

stroke_df[stroke_df.Residence_type.=="Rural", :Residence_type] .= "1"
stroke_df[stroke_df.Residence_type.=="Urban", :Residence_type] .= "2"
stroke_df[!, :Residence_type] = parse.(Int32, stroke_df[!, :Residence_type])

stroke_df[stroke_df.smoking_status.=="formerly smoked", :smoking_status] .= "1"
stroke_df[stroke_df.smoking_status.=="never smoked", :smoking_status] .= "0"
stroke_df[stroke_df.smoking_status.=="smokes", :smoking_status] .= "2"
stroke_df[!, :smoking_status] = parse.(Int32, stroke_df[!, :smoking_status])

unique.(eachcol(stroke_df))

CSV.write("./stroke_dataset_no_bmi.csv", stroke_df_no_bmi)