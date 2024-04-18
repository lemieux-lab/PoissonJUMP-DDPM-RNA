#! Also the other smaller stuff like lr. 

import Pkg
Pkg.activate("src/.")


using Random
using Distributions
using StatsBase
using Statistics
using Flux
include("conditional_mlp")
include("samplings.jl")


using Printf
using ProgressBars


function linear_beta_schedule(num_timesteps::Int, β_start=0.0001f0, β_end=0.02f0)
    scale = convert(typeof(β_start), 1000 / num_timesteps)
    β_start *= scale
    β_end *= scale
    range(β_start, β_end; length=num_timesteps)
end

# Important DDPM constants
betas = linear_beta_schedule(1000,0.001, 0.0845)
betas = collect(betas)
alphas = sqrt.(cumprod(1.0 .- betas, dims=1)) 
alphas_prev = cat([1.0], alphas[1:end-1], dims=1)

deltas = (alphas_prev .- alphas) * 100


function gradient_step(X, Q, timesteps, model, opt, loss)

    ps=Flux.params(model)

    grads = Flux.gradient(ps) do
        loss(model, X, Q, timesteps)
    end

    Flux.update!(opt, ps, grads)
    train_l_tmp = loss(model, X, Q, timesteps)
    return train_l_tmp
end


function l2_penalty(model)
    
    sum_w=0
    for param in Flux.params(model)
        sum_w+=sum(abs2,param)
    end

    return sum_w
end

# Train one fold
function train_fold_binom(data; lmb=10, T=500, 
                                mode="binomial", time_choice="random", 
                                model_type="DDPM", n_epochs=2, 
                                batch_size=32, test_size=100, lr=0.0001)
    n_samples = size(data)[1]
    in_dim = size(data)[2]
    #! Make sure to assign device (gpu/cpu)
    data_ori = data
    data = Float32.(log10.(data.+1))

    shuffled_ids = shuffle(collect(1:n_samples))
    test_ids = shuffled_ids[1:test_size]
    train_ids = shuffled_ids[test_size+1:end]
    n_batches = Int(ceil(length(train_ids)/batch_size))

    t_fixed = T

    opt = ADAM(lr)
    

    #! Possible loss functions that can be used
    #loss(model, x, q, timesteps) = Flux.Losses.mse(gpu(deltas[timesteps]').*model(gpu(q'), timesteps'), gpu(deltas[timesteps]'.*x'))  #+ 1e-3*l2_penalty(model)
    #loss(model, x, q, timesteps) = Flux.Losses.huber_loss(gpu(deltas[timesteps]').*model(gpu(q'), timesteps'), gpu(deltas[timesteps]'.*x'))  #+ 1e-3*l2_penalty(model)
    loss(model, x, q, timesteps) =  mean(poisson_kl(gpu(deltas[timesteps] .* x), gpu(deltas[timesteps]) .* model(gpu(q'), timesteps')')) #+ 1e-4*l2_penalty(model)

    #! WARNING: the gpu is hard coded at the moment. This will need to be fixed.
    model = gpu(Conditional_MLP(Dict("in_dim" => in_dim, "base_dim" => 2000)))

    train_x = data[train_ids, :]
    test_x_ori  = data_ori[test_ids, :]
    test_x  = log10.(test_x_ori .+1 )

    train_loss = []
    test_loss  = []

    for e in 1:n_epochs
        epoch_shuffle=shuffle(train_ids)

        for b in 1:n_batches
            # Batch idx
            if b == n_batches
                batch_idx= epoch_shuffle[(b-1)*batch_size+1:end]
            else
                batch_idx = epoch_shuffle[(b-1)*batch_size+1:b*batch_size]
            end   

            batch_x = data[batch_idx, :]

            if time_choice == "random"
                timesteps = reshape(rand(1:t_fixed, length(batch_idx)), :, 1)
                timesteps_test = reshape(rand(1:t_fixed, length(test_ids)), :, 1)

            elseif time_choice == "fixed"
                timesteps = reshape(repeat([t_fixed], length(batch_idx)), :, 1)
                timesteps_test = reshape(repeat([t_fixed], length(test_ids)), :, 1)

            end
 
            if mode=="poisson"
                Q = log10.(q_sample(data_ori[batch_idx,:], timesteps, alphas) .+1 ) 
                Q_test = log10.(q_sample(test_x_ori, timesteps_test, alphas) .+1 ) 

            else
                Q = log10.(q_sample_binom(data_ori[batch_idx,:], alphas[timesteps], alphas) .+1 )
                Q_test = log10.(q_sample_binom(test_x, alphas[timesteps_test], alphas) .+1 )
            end

            train_l_tmp = gradient_step(batch_x, Q, timesteps, model, opt,loss)
            push!(train_loss, train_l_tmp)


            test_l_tmp = loss(model, test_x, Q_test,timesteps_test)
            push!(test_loss, test_l_tmp)
        end
    end

    return model, train_loss, test_loss, train_ids, test_ids

end

### Example usage
### Replace data_matrix with samples x features matrix. 
# trained_model, train_loss, test_loss, train_idx, test_idx = train_fold_binom(data_matrix;
#                                                lmb=10, T=600, 
#                                                time_choice="random", model_type="DDPM",mode="poisson",
#                                                n_epochs=5)


# The iterative denoising
# Q: noisy data
# T: starting timestep (integer)
# Trained model: Trained DDPM model.
# FIXME: this works poorly for large datasets. 
function iter_denoise(Q, T, trained_model)
    Tim = repeat([T], size(Q)[1])
    z_t = log10.(q_sample(Q, Tim, alphas) .+1 )

    for (frame, i) in enumerate(collect(T:-1:2))
        Tim = repeat([i], length(samples))
        
        x_hat = cpu(trained_model(gpu(z_t'), Tim')')

        # To compensate for log-normalisation of gene expression counts
        adj_z_t = clamp.(10 .^ (z_t) .- 1, 0, Inf)
        adj_x_hat = clamp.(10 .^ (x_hat) .- 1, 0, Inf)
        
        p_t = (alphas[i-1] - alphas[i])
        
        rate = adj_x_hat.*p_t

        z_t = log10.(adj_z_t + [i[1] for i in rand.(Distributions.Poisson.(rate), 1)].+1)
    end
    return z_t
end