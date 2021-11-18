require "csv"
require "evolvenet"

# data structures to hold the input and results
inputs = Array(Array(Float64)).new
outputs = Array(Array(Float64)).new

# read the file
raw = File.read("./data/weight-height.csv")
csv = CSV.new(raw, headers: true)

# load the data structures
while (csv.next)
  inputs << [csv.row["Height"].to_f64]
  outputs << [csv.row["Weight"].to_f64]
end

# normalize the data
training = EvolveNet::TrainingData.new(inputs, outputs)
training.normalize_min_max

# create a network
model : EvolveNet::Network = EvolveNet::Network.new
model.add_layer(:input, 1, :none)
model.add_layer(:hidden, 1, :sigmoid)
model.add_layer(:output, 1, :none)
model.fully_connect

# evolve the network
organism = EvolveNet::Organism.new(model, 10, 10)
model = organism.evolve(training.data, 1000)

results = Array(Float64).new
results << model.run([0])[0].to_f64.round(8)
results << model.run([0.25])[0].to_f64.round(8)
results << model.run([0.5])[0].to_f64.round(8)
results << model.run([0.75])[0].to_f64.round(8)
results << model.run([1])[0].to_f64.round(8)

puts results
# results = model.run(training.normalize_inputs([75]))
# puts training.denormalize_outputs(results)
