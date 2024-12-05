module MiniNN

include("engine.jl")
include("nn.jl")

export Value, backward!, gradient, Neuron, ReLU, Linear, MLP, get_parameters, zero_gradients, o

end # module MiniNN
