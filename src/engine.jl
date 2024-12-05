using Base

const DEBUG = false

Base.@kwdef mutable struct Value{T <: Float64}
	x::T = zero(T)
	∂x::T = zero(T)
	backward::Function = () -> nothing
	prev::Set{Value{T}} = Set{Value{T}}()
	op::String = ""
end
Value(x::T) where {T <: Float64} = Value{Float64}(x = x)
Value(x::T) where {T <: Int64} = Value{Float64}(x = float(x))
Value(x::T, children::Vector{Value{T}}, op::String) where {T <: Float64} = Value{Float64}(x = x, prev = Set(children), op = op)

function backward!(node::Value)
	topo = Value[]
	visited = Set{Value}()

	function build_topo!(v)
		v ∈ visited && return
		push!(visited, v)
		foreach(build_topo!, v.prev)
		push!(topo, v)
	end

	build_topo!(node)
	node.∂x = 1
	foreach(n -> n.backward(), reverse(topo))
end
backward!(node::Number) = backward!(Value(node))

function gradient(loss::Value{Float64}, parameters::Vector{Value{Float64}})
	backward!(loss)
	map(p -> p.∂x, parameters)
end

function gradient(expr::Function, point::Tuple{Vararg{Value}})
	y = expr(point...)

	backward!(y)

	map(n -> n.∂x, point)
end
gradient(expr::Function, point::Tuple{Vararg{Number}}) = gradient(expr, map(Value, point))
gradient(expr::Function, point::Value) = gradient(expr, (point,))
gradient(expr::Function, point::Number) = gradient(expr, (Value(point),))

function Base.:+(l::Value, r::Value)::Value
	out = Value(l.x + r.x, [l, r], "+")

	out.backward = () -> begin
		l.∂x += out.∂x
		r.∂x += out.∂x
	end

	if DEBUG
		println(l.x, " + ", r.x, " = ", out.x)
	end

	out
end
Base.:+(l::Value, r::Number)::Value = l + Value(float(r))
Base.:+(l::Number, r::Value)::Value = Value(float(l)) + r


function Base.:*(l::Value, r::Value)::Value
	out = Value(l.x * r.x, [l, r], "*")

	out.backward = () -> begin
		l.∂x += r.x * out.∂x
		r.∂x += l.x * out.∂x
	end

	if DEBUG
		println(l.x, " * ", r.x, " = ", out.x)
	end

	out
end
Base.:*(l::Number, r::Value)::Value = Value(float(l)) * r
Base.:*(l::Value, r::Number)::Value = l * Value(float(r))

function Base.:^(base::Value, exponent::Float64)::Value
	out = Value(base.x^exponent, [base], "^")

	out.backward = () -> begin
		base.∂x += (exponent * base.x^(exponent - 1)) * out.∂x
	end

	if DEBUG
		println(base.x, " ^ ", exponent, " = ", out.x)
	end

	out
end
Base.:^(base::Value, exponent::Int64)::Value = Base.:^(base, float(exponent))


function Base.:inv(n::Value)::Value

	out = Value(1.0 / n.x, [n], "inv")
	out.backward = () -> begin
		n.∂x += (-1 * 1 / (n.x^2)) * out.∂x
	end

	if DEBUG
		println("1.0 / ", n.x, " = ", out.x)
	end

	out
end

Base.:-(l::Value, r::Value)::Value = l + -1.0 * r
Base.:-(l::Value, r::Number)::Value = l - Value(float(r))
Base.:-(l::Number, r::Value) = Value(float(l)) - r


Base.:/(t::Value, b::Value)::Value = t * (b^-1)
Base.:/(t::Number, b::Value)::Value = Value(float(t)) / b
Base.:/(t::Value, b::Number)::Value = t / Value(float(b))


"""
ReLU activation function
"""
function ReLU(n::Value)
	out = Value(if (n.x > 0)
		n.x
	else
		0.0
	end, [n], "relu")

	out.backward = () -> begin
		n.∂x += (
			if (n.x > 0)
				1.0
			else
				0.0
			end
		) * out.∂x
	end

	if DEBUG
		println("relu(", n.x, ") = ", out.x)
	end

	out
end
ReLU(n::Number) = ReLU(Value(n))
