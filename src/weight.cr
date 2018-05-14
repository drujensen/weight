require "csv"
require "shainet"

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
training = SHAInet::TrainingData.new(inputs, outputs)
training.normalize_min_max

# create a network
model : SHAInet::Network = SHAInet::Network.new
model.add_layer(:input, 1, :memory, SHAInet.none)
model.add_layer(:hidden, 1, :memory, SHAInet.sigmoid)
model.add_layer(:output, 1, :memory, SHAInet.none)
model.fully_connect

# train the network
model.train(training.data, :sgdm, :mse, 1000, -1.0)

results = model.run(training.normalize_inputs([75]))
puts training.denormalize_outputs(results)
