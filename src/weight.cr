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
model.learning_rate = 0.01
model.train(training.data, :sgdm, :mse, 200, -1.0, 10)

results = Array(Float64).new
results << model.run([0])[0].round(8)
results << model.run([0.25])[0].round(8)
results << model.run([0.5])[0].round(8)
results << model.run([0.75])[0].round(8)
results << model.run([1])[0].round(8)

puts results
# results = model.run(training.normalize_inputs([75]))
# puts training.denormalize_outputs(results)
