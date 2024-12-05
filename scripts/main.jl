using MiniNN: Linear, MLP, Value, backward!, get_parameters, gradient, zero_gradients, optimizer_step

# define a very simple MLP with randomly initialized parameters
network = MLP([Linear(1, 5), Linear(5, 5), Linear(5, 1)])

# i am lazy so we only try to learn one sample
# no batching yet
y_true = Value(0.5)
x = [Value(0.5)]

for epoch in 1:100

    # get nn prediction
    y = network(x)
    
    # calculate MSE loss
    loss = (y - y_true)^2

    # clear the gradients from previous epoch
    zero_gradients(network)

    # do the backward pass (exciting bit)
    backward!(loss)

    # update the nn parameters
    optimizer_step(network)

    if epoch % 10 ==0
        println("Epoch = ", epoch, ", Loss = ", loss.x)        
    end
end