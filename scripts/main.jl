using MiniNN: Linear, MLP, Value, backward!, get_parameters, gradient, zero_gradients, optimizer_step

network = MLP([Linear(1, 5), Linear(5, 5), Linear(5, 1)])

y_true = Value(0.5)
x = [Value(0.5)]

for epoch in 1:100

    y = network(x)
    
    loss = (y - y_true)^2

    zero_gradients(network)
    backward!(loss)
    optimizer_step(network)

    if epoch % 10 ==0
        println("Epoch = ", epoch, ", Loss = ", loss.x)        
    end
end