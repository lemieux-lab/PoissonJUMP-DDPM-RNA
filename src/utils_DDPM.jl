# All the packages
using CSV
using DataFrames
using Random
using Distributions
using StatsBase
using Statistics
using Flux
using Printf
using ProgressBars



## KL_loss for poisson-ddpm
function poisson_kl(x, y, eps = 1e-12)
    return x .* (log.(clamp.(x, eps, Inf)) .- log.(clamp.(y, eps,Inf))).-(x.-y)
end


## Choose between huber-loss, MSE or poisson-kl. Also if the model uses timsteps or not
function choose_loss(loss_name = "huber", time_choice="random")
    losses = Dict("huber" => Flux.Losses.huber_loss, "MSE" => Flux.Losses.mse, "poisson_kl" => poisson_kl)
    loss_fct = losses[loss_name]

    if time_choice == "random" || time_choice == "batch_random" 
        loss  = (model, x, q, timesteps) -> loss_fct(gpu(deltas[timesteps]').*model(gpu(q'), timesteps'), gpu(deltas[timesteps]'.*x'))
        return loss
    else
       loss = (model, x, q, timesteps) -> loss_fct(model(gpu(q')), gpu(x')) 
       return loss
    end

    return loss
end

## Currently, chooses between a DDPM and a skip-connection auto-encoder
function choose_model(model_type, in_dim; inter_dim=10000)
    if model_type=="DDPM"
        return Conditional_MLP(Dict("in_dim" => in_dim, "base_dim" => inter_dim, "temb_dim" => 256))

    elseif model_type=="skip-ae"
        return SkipConnection(
            Flux.Chain(Flux.Dense(in_dim, inter_dim),
                SkipConnection(
                    Flux.Chain(Flux.Dense(inter_dim,inter_dim),Flux.Dense(inter_dim,inter_dim)),
                +),
                Flux.Dense(inter_dim,in_dim)),
            +)
    end
    return nothing
end

