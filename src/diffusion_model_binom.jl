#! Also the other smaller stuff like lr. 

import Pkg
Pkg.activate("src/.")
include("utils_DDPM.jl")
include("conditional_mlp.jl")
include("samplings.jl")


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
# NOTE: this code will require a GPU
function train_fold_binom(data; lmb=10, t_fixed=500, mode="binomial", time_choice="random", 
    model_type="DDPM", loss_name="huber", n_epochs=2, test_size=32, batch_size=32)

    n_samples = size(data)[1]
    in_dim = size(data)[2]

    data_ori = data
    data = Float32.(log10.(data.+1))
    data_scale = data

    shuffled_ids = shuffle(collect(1:n_samples))
    test_ids = shuffled_ids[1:test_size]
    train_ids = shuffled_ids[test_size+1:end]

    n_batches = Int(ceil(length(train_ids)/batch_size))

    opt = ADAM(1e-4)
    loss = choose_loss(loss_name, time_choice)

    model = gpu(choose_model(model_type, in_dim))



    train_x = data[train_ids, :]
    test_x_ori  = data_ori[test_ids, :]
    test_x = log10.(test_x_ori .+1 )

    train_loss = []
    test_loss  = []


    for e in ProgressBar(1:n_epochs)
        epoch_shuffle=shuffle(train_ids)

        for b in 1:n_batches
            # Batch idx
            if b == n_batches
                batch_idx= epoch_shuffle[(b-1)*batch_size+1:end]
            else
                batch_idx = epoch_shuffle[(b-1)*batch_size+1:b*batch_size]
            end   

            batch_x = data[batch_idx, :]
            batch_x_scale = data_scale[batch_idx, :]

            # If we have a model training for all coverages or only a specific coverage
            if time_choice == "random"
                # Selects a random t for all samples in batch
                timesteps = reshape(rand(1:t_fixed, length(batch_idx)), :, 1)
                timesteps_test = reshape(rand(1:t_fixed, length(test_ids)), :, 1)

            elseif time_choice == "batch_random"
                # Selects the same random t for all samples in batch 
                timesteps = reshape(repeat(rand(1:t_fixed, 1), length(batch_idx)), :, 1)
                timesteps_test = reshape(repeat(rand(1:t_fixed, 1), length(test_ids)), :, 1)

            elseif time_choice == "fixed"
                # Awlays selects the t_fixed for t
                timesteps = reshape(repeat([t_fixed], length(batch_idx)), :, 1)
                timesteps_test = reshape(repeat([t_fixed], length(test_ids)), :, 1)
            end
            alphas_t = alphas[timesteps]
            alphas_t_test = alphas[timesteps_test]

            # Allowing for discrete or continuous sampling
            if mode=="poisson"
                Q = log10.(q_sample(data_ori[batch_idx,:], timesteps, alphas) .+1 )
                Q_test = log10.(q_sample(test_x_ori, timesteps_test, alphas)  .+1 )

            else mode=="binom"
                Q = log10.(q_sample_binom(data_ori[batch_idx,:], alphas[timesteps]) .+1 )
                Q_test = log10.(q_sample_binom(test_x_ori, alphas[timesteps_test]) .+1 )
            end

            train_l_tmp = gradient_step(batch_x_scale, Q, timesteps, model, opt,loss)
            test_l_tmp = loss(model, test_x, Q_test, timesteps_test)

            push!(train_loss, train_l_tmp)
            push!(test_loss, test_l_tmp)

        end
    end
    return model, train_loss, test_loss, train_ids, test_ids

end


### Example usage
### Replace data_matrix with samples x features matrix. 
### See utils_DDPM.jl for possible models and losses
#trained_model, train_loss, test_loss, train_idx, test_idx = train_fold_binom(data_matrix;
#                                                lmb=10, t_fixed=800, 
#                                                time_choice="random", model_type="DDPM",mode="binom",
#                                                n_epochs=10, loss_name="huber")

# The iterative denoising
# Q: noisy data
# T: starting timestep (integer)
# Trained model: Trained DDPM model.
# NOTE: this works poorly for large datasets. 
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