using Distributions

### Original poisson q_sample
function q_sample(x_0, t, alphas; λ = 1)
    rate = alphas[t] * λ # * diffusion.lbd
    z_t = [i[1] for i in rand.(Distributions.Poisson.(rate .* x_0), 1)]
    
    # Rescaling?
    return z_t./λ
end

### Binomial version of q_sample
function q_sample_binom(x_0, p, alphas; λ = 10000)
    x_0 = round.(x_0)
    z_t = [i[1] for i in rand.(Distributions.Binomial.(x_0, p), 1)] #+ rand.(Normal(0,20), size(x_0)[1], size(x_0)[2])
   
    return z_t 
end 

### Same as q_sample, but prob is calculated beforehand
function q_sample_poiss_approx(x_0, p, alphas; λ = 10000)
    z_t = [i[1] for i in rand.(Distributions.Poisson.(x_0.*p), 1)] #+ rand.(Normal(0,20), size(x_0)[1], size(x_0)[2])
    return z_t #./λ 
end 

function poisson_kl(x, y, eps = 1e-12)

    return x .* (log.(clamp.(x, eps, Inf)) .- log.(clamp.(y, eps,Inf))).-(x.-y)
end
