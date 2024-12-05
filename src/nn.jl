using Random: rand

struct Neuron
	w::Array{Value{Float64}}
	b::Value{Float64}
	σ::Function

	function Neuron(in_features::Integer, σ::Function)
		w = [Value(2 * rand(Float64) - 1) for _ ∈ 1:in_features]
		b = Value(rand(Float64))
		new(w, b, σ)
	end
end

(n::Neuron)(x::Vector) = n.σ(sum(wi * xi for (wi, xi) in zip(n.w, x)) + n.b)

struct Linear
	neurons::Array{Neuron}
	σ::Function

	function Linear(in_features::Integer, out_features::Integer, σ::Function = ReLU)
		new([Neuron(in_features, σ) for _ ∈ 1:out_features], σ)
	end
end

(l::Linear)(x::Vector) = [n(x) for n in l.neurons]

struct MLP
	layers::Array{Linear}
end

function (network::MLP)(x::Vector)
	for layer in network.layers
		x = layer(x)
	end
	length(x) > 1 ? x : x[1]
end

get_parameters(neuron::Neuron) = [neuron.w..., neuron.b]
get_parameters(linear::Linear) = [p for n in linear.neurons for p in get_parameters(n)]
get_parameters(network::MLP) = [p for layer in network.layers for p in get_parameters(layer)]

function zero_gradients(network::MLP)
	for p in get_parameters(network)
		p.∂x = 0
	end
end

optimizer_step(network::MLP, lr::Float64 = 0.01) = map(p -> p.x -= lr * p.∂x, get_parameters(network))
