import Pkg
Pkg.activate("/home/golem/scratch/munozc/Julia-try/pair_prog/")


using CSV
using DataFrames
using Random
using Distributions
using StatsBase
using Statistics

include("conditional_mlp")
include("samplings.jl")

using CairoMakie
using Plots
using Flux
using TSne

using Printf
using ProgressBars

function linear_beta_schedule(num_timesteps::Int, β_start=0.0001f0, β_end=0.02f0)
    scale = convert(typeof(β_start), 1000 / num_timesteps)
    β_start *= scale
    β_end *= scale
    range(β_start, β_end; length=num_timesteps)
end

betas = linear_beta_schedule(1000,0.001, 0.0845)
betas = collect(betas)
alphas = sqrt.(cumprod(1.0 .- betas, dims=1)) 
alphas_prev = cat([1.0], alphas[1:end-1], dims=1)

deltas = (alphas_prev .- alphas) * 100

### Gene counts
data_matrix_1 = Array(CSV.read("/home/golem/scratch/munozc/DDPM/tx_counts_1pct_cov.csv", DataFrame, header=false))'
data_matrix_1_log=log10.(data_matrix_1.+1)
data_matrix = data_matrix_1_log

data_matrix_10 = Array(CSV.read("/home/golem/scratch/munozc/DDPM/tx_counts_10pct_cov.csv", DataFrame, header=false))'
data_matrix_10_log = log10.(data_matrix_10 .+1)

data_matrix_100 = Array(CSV.read("/home/golem/scratch/munozc/DDPM/tx_counts_100pct_cov.csv", DataFrame, header=false))'
data_matrix_100_log = log10.(data_matrix_100 .+1)
dm_tmp100_log_1 = data_matrix_100_log
data_matrix=data_matrix_100_log

dm_tmp1 = Array(CSV.read("/home/golem/scratch/munozc/data_1pct_bams/naive_unpaired_counts_all.csv", DataFrame, header=false))'
dm_tmp1_log = log10.(dm_tmp1.+1)


dm_tmp10 = Array(CSV.read("/home/golem/scratch/munozc/naive_unpaired_counts_all_10pct.csv", DataFrame, header=false))'
dm_tmp10_log = log10.(dm_tmp100.+1)

dm_tmp100 = Array(CSV.read("/home/golem/scratch/munozc/naive_unpaired_counts_all_100pct.csv", DataFrame, header=false))'
dm_tmp100_log = log10.(dm_tmp100.+1)

mean(sum(dm_tmp100_1, dims=2))

most_var = sortperm(var(dm_tmp100_log, dims=1)[1,:], rev=true)[1:1000]
avg = mean(dm_tmp100_log[:,most_var], dims=1)
avgs = repeat(avg, size(dm_tmp100_log[:,most_var])[1])
cor(dm_tmp100_log[:,most_var], avgs, dims=2)
cor(dm_tmp100_log, dims=2)



timesteps = repeat([455], size(dm_tmp100)[1])
Q_test = q_sample(dm_tmp100, timesteps, alphas,λ=1)
dm_tmp100
Q_test1 = log10.(q_sample_poiss_approx(dm_tmp100, 0.01, alphas).+1)
Q_test2 = q_sample_poiss_approx(dm_tmp100, 0.01, alphas)
dm_tmp1
all_Qs = []
for i in ProgressBar(alphas[1:10:end])
    Q = q_sample_binom(dm_tmp100, i, alphas)
    #noises = Matrix(mapreduce(permutedims, vcat, rand.(Normal.(0,sig), size(data_matrix)[1])[1,:])')
    #Q = log.(clamp.(Q + noises, 0, Inf).+1) 
    push!(all_Qs, Q)
end

maximum(data_matrix_1[2,:])
data_m
points = Observable(vec(log10.(all_Qs[1].+1)))
points = Observable(vec(log10.(Q_test.+1)).+2)
perc = Observable(1-0.01*(1-1))

#Axis(f[1,1], aspect = DataAspect())
#ax = Axis(f[1,1])
alphas[455]
## ; Spearman: $(@sprintf("%.4f", corel2[]))
f,ax,hm= CairoMakie.hexbin(vec(dm_tmp100_log), points, colorscale=log10,cellsize=0.1, 
        aspect = AxisAspect(1), figure=(;fontsize=20), 
        axis=(ylabel="Noised log-counts",
            xlabel="100% coverage log-counts", 
            title="Percentage: 0.001"))

f,ax,hm= CairoMakie.hexbin(vec(dm_tmp100_log), vec(log10.(Q_test1.+1)), colorscale=log10,cellsize=0.1, 
    aspect = AxisAspect(1), figure=(;fontsize=20), 
    axis=(ylabel="Noised log-counts",
        xlabel="100% coverage log-counts", 
        title="Percentage: 0.01"))

#Axis(f[1,1], xlabel="test")
CairoMakie.ablines!( 0,1, color=[:red], "t=0", linestyle=:dash)
#CairoMakie.ylims!(ax, (0,11))
#ax

#CairoMakie.scatter!(ax, dm_vec, result_vec)
CairoMakie.Colorbar(f[1,2], hm,
    label = "Number of expressions",
    height = Relative(0.5)
)
f

collect(1:-0.01:0)
frames = 1:100
record(f, "test_nonoise_fixed_again.mp4", frames;
        framerate = 30) do frame
    perc[] = alphas[1:10:end][frame]
    ax.title="Percentage:  $(perc[])"
    points[] = vec(log10.(all_Qs[frame].+1))
end
alphas[600]

Random.seed!(1234)
using Flux.Optimise

function gradient_step(X, Q, timesteps, model, opt, loss)

    ps=Flux.params(model)
    
    #println(size(X))
    #println(size(Q))
    # We gave matrix here 
    grads = Flux.gradient(ps) do
        loss(model, X, Q, timesteps)

        #loss(model, X, Q)
    end
    #println(minimum(model(Q,timesteps)))
    println("Train")
    #for i in grads
    #    println(i)
    #end
    #println(grads)
    #println(loss(model, X, Q, timesteps))
    

    #println(ps)
    Flux.update!(opt, ps, grads)

    train_l_tmp = loss(model, X, Q, timesteps)
    #train_l_tmp = loss(model, X, Q)
    println(train_l_tmp)
    
    #println(ps)
    #println(model(Q,timesteps)')
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
function train_fold_binom(data; lmb=10, T=500, mode="binomial", time_choice="random", model_type="DDPM", n_epochs=2)
    n_samples = size(data)[1]
    in_dim = size(data)[2]
    # gpu
    data_ori = data
    data = Float32.(log10.(data.+1))
    #n_epochs = 4
    #FIXME: don't hardcode this

    test_size = 100
    # Test size : 
    shuffled_ids = shuffle(collect(1:n_samples))
    test_ids = shuffled_ids[1:test_size]
    train_ids = shuffled_ids[test_size+1:end]


    batch_size = 32
    n_batches = Int(ceil(length(train_ids)/batch_size))
    #Int(ceil(size(data)[1]/batch_size))
    #batch_size = 32

    t_fixed = T



    #println(n_samples)
    #println(in_dim)

    #shuffled_ids = shuffle(collect(1:n_samples))

    #opt =  Optimiser(ClipValue(1e-3), ADAM(0.0001))
    opt = ADAM(0.0001)
    
    ### Toying with losses
    #FIXME: I should probably change this. 
    #if model_type == "ae"
        #loss(model, x, q, timesteps) = Flux.Losses.huber_loss(model(q'), x')
        #loss(model, x, q, timesteps) = Flux.Losses.mse(model(q'), x')
        #model = Chain(SkipConnection(Chain(Dense(size(data)[2] => 2000, leakyrelu),
        #        Dense(2000 => size(data)[2])),+), relu)
        #model = Chain(SkipConnection(Chain(Dense(size(data)[2] => size(data)[2], relu)),+), relu)
        #model = Chain(Dense(size(data)[2] => 2000, leakyrelu), Dense(2000 => size(data)[2], relu))
        #model=Flux.Chain(Flux.Dense(size(data)[2] => size(data)[2], relu), Flux.Dense(size(data)[2] => size(data)[2], relu))
        #model = gpu(model)
    #else
        #loss(model, x, q, timesteps) = Flux.Losses.huber_loss(deltas[timesteps]'.*model(q', timesteps'), deltas[timesteps]'.*x')  #+ 1e-3*l2_penalty(model)
        #loss(model, x, q, timesteps) = Flux.Losses.mse(deltas[timesteps]'.*model(q', timesteps'), deltas[timesteps]'.*x')  #+ 1e-3*l2_penalty(model)
        
        #loss(model, x, q, timesteps) = Flux.Losses.mse(gpu(deltas[timesteps]').*model(gpu(q'), timesteps'), gpu(deltas[timesteps]'.*x'))  #+ 1e-3*l2_penalty(model)
        #loss(model, x, q, timesteps) = Flux.Losses.huber_loss(gpu(deltas[timesteps]').*model(gpu(q'), timesteps'), gpu(deltas[timesteps]'.*x'))  #+ 1e-3*l2_penalty(model)
        #loss(model, x, q, timesteps) = Flux.Losses.mse(model(gpu(q'), timesteps'), gpu(x'))  #+ 1e-3*l2_penalty(model)
        #loss(model, x, q, timesteps) = Flux.Losses.huber_loss(model(gpu(q'), timesteps'), gpu(x'))  #+ 1e-3*l2_penalty(model)
        
        #loss(model, x, q, timesteps) = Flux.Losses.huber_loss(model(q', timesteps'), x') #+ 1e-5*l2_penalty(model)
        #loss(model, x, q, timesteps) = Flux.Losses.mse(model(q', timesteps'), x')#+ 1e-2*l2_penalty(model)
        #loss(model, x, q, timesteps) = Flux.Losses.mse(model(q'), x') #+ 1e-2*l2_penalty(model)
        #loss(model, x, q) = Flux.Losses.huber_loss(model(q'), x')
        #loss(model, x, q, timesteps) =  mean(poisson_kl(deltas[timesteps] .* x, deltas[timesteps] .* model(q', timesteps')')) #+ 1e-4*l2_penalty(model)
        loss(model, x, q, timesteps) =  mean(poisson_kl(gpu(deltas[timesteps] .* x), gpu(deltas[timesteps]) .* model(gpu(q'), timesteps')')) #+ 1e-4*l2_penalty(model)
        #loss(model, x, q, timesteps) =  mean(poisson_kl(gpu( x), model(gpu(q'), timesteps')')) #+ 1e-4*l2_penalty(model)
        

        #loss(model, x, q, timesteps) =  mean(poisson_kl(x,model(q', timesteps')')) #+ 1e-2*l2_penalty(model)
        #loss(model, x, q, timesteps) =  mean(poisson_kl(x,model(gpu(q'), timesteps')')) #+ 1e-2*l2_penalty(model)

        model = gpu(Conditional_MLP(Dict("in_dim" => in_dim, "base_dim" => 2000)))
        #model = Flux.Scale(in_dim, bias=true)
    #end

    #train_idx = train_ids[1:batch_size*n_batches]
    #test_idx  = train_ids[batch_size*n_batches+1:end]    
    
    train_x = data[train_ids, :]
    test_x_ori  = data_ori[test_ids, :]
    test_x  = log10.(test_x_ori .+1 )

    train_loss = []
    test_loss  = []

    #TODO: losses and correlations lists 
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

            # FIXME: training size and testing size

            if time_choice == "random"
                timesteps = reshape(rand(1:t_fixed, length(batch_idx)), :, 1)
                #timesteps = reshape(repeat([t_fixed], 20), :, 1)
                timesteps_test = reshape(rand(1:t_fixed, length(test_ids)), :, 1)
            elseif time_choice == "fixed"
                timesteps = reshape(repeat([t_fixed], length(batch_idx)), :, 1)
                timesteps_test = reshape(repeat([t_fixed], length(test_ids)), :, 1)
            end
            ## FIXME: force a scaling that kinda works.
            # const_train =  
            # const_test = 


            #### NOTE: this is currently a sample then log
            #### Previously, I did log then sample...
            #### Math may not be exact. 
            if mode=="poisson"
                Q = log10.(q_sample(data_ori[batch_idx,:], timesteps, alphas) .+1 ) 
                #Q = q_sample(data_ori[batch_idx,:], timesteps, alphas)
                #Q = log10.(Q*(1/alphas[455]) .+ 1)  
                
                Q_test = log10.(q_sample(test_x_ori, timesteps_test, alphas) .+1 ) 
                #Q_test = q_sample(test_x, timesteps_test, alphas)
                #Q_test = log10.(Q_test*(1/alphas[455]) .+ 1)
                
            else
                Q = log10.(q_sample_binom(data_ori[batch_idx,:], alphas[timesteps], alphas) .+1 ) #.+cons[batch_idx,:]
                Q_test = log10.(q_sample_binom(test_x, alphas[timesteps_test], alphas) .+1 ) #.+cons[test_idx,:]
            end

            #train_l_tmp = gradient_step(gpu(batch_x), gpu(Q), gpu(timesteps), model, opt,loss)
            train_l_tmp = gradient_step(batch_x, Q, timesteps, model, opt,loss)
            push!(train_loss, train_l_tmp)


            #FIXME: test loss has an error somewhere
            println("Test")
            ###
            #test_l_tmp = loss(model, gpu(test_x), gpu(Q_test), gpu(timesteps_test))
            test_l_tmp = loss(model, test_x, Q_test,timesteps_test)
            #test_l_tmp = loss(model, test_x, Q_test)
            println(test_l_tmp)
            push!(test_loss, test_l_tmp)

            println()

            #break
        end
        #break
    end

    return model, train_loss, test_loss, train_ids, test_ids

end

size(dm_tmp100)

model = gpu(Conditional_MLP(Dict("in_dim" => 19962, "base_dim" => 2000)))
model = Conditional_MLP(Dict("in_dim" => 19962, "base_dim" => 2000))

timesteps = reshape(rand(1:600, 100), :, 1)
timesteps_g = gpu(timesteps')
testy = log10.(q_sample(dm_tmp100[1:100,:], timesteps, alphas) .+1 )
testy_g = gpu(testy')

model(gpu(testy'), timesteps')

Flux.Chain([Conditional_layer(100,100,100) for r in 1:5]...)

model
isbitstype(typeof(model.layers))
include("conditional_mlp")


replace_nan(v) = map(x -> isnan(x) ? zero(x) : x, v)

function var_make(mat)
    tmp = mat .- mean(mat, dims=1)
    return tmp
    #min_tmp = minimum(tmp, dims=1)
    #max_tmp = maximum(tmp, dims=1)

    #return replace_nan((tmp .- min_tmp) ./ (max_tmp .- min_tmp))
end

tmp = dm_tmp100_log .- mean(dm_tmp100_log, dims=1)

min_tmp = minimum(tmp, dims=1)
max_tmp = maximum(tmp, dims=1)


# FIXME: 
adj_dm = (tmp .- min_tmp) ./ (max_tmp .- min_tmp)
adj_dm = replace_nan(adj_dm)
adj_dm

sort_idx = sortperm(cancer_types[bigs])

cors = cor(adj_dm[sort_idx,:], dims=2)
cors = cor(dm_tmp100_log[sort_idx,:], dims=2)

CairoMakie.heatmap(cors)


using CUDA
CUDA.memory_status()
GC.gc()
draw_network(model, save_to="test_network.png")
dm_tmp100
most_var = sortperm(var(dm_tmp100_log, dims=1)[1,:], rev=true)[1:1000]
trained_model, train_loss, test_loss, train_idx, test_idx = train_fold_binom(dm_tmp100[:,most_var];
                                                lmb=10, T=600, 
                                                time_choice="random", model_type="DDPM",mode="poisson",
                                                n_epochs=2)
trained_model, train_loss, test_loss, train_idx, test_idx = train_fold_binom(dm_tmp100;
                                                lmb=10, T=600, 
                                                time_choice="random", model_type="DDPM",mode="poisson",
                                                n_epochs=5)
trained_model, train_loss, test_loss, train_idx, test_idx = train_fold_binom(dm_tmp100; 
                                                lmb=10, T=559, 
                                                time_choice="fixed", model_type="DDPM", mode="poisson",
                                                n_epochs=10)
dm_tmp100_log
trained_model, train_loss, test_loss, train_idx, test_idx = train_fold_binom(dm_tmp100; 
                                                lmb=10, T=1000, mode="poisson")
Q_test = dm_tmp1_log[test_idx,:]

testmode!(trained_model)

dm_tmp100

dm_tmp100_log
trained_model = cpu(trained_model)

sort_idx = sortperm(cancer_types[bigs])
alphas[559]*100
t_fixed=723
t_fixed=646
t_fixed=559
t_fixed=455
t_fixed=100
#timesteps = reshape(rand(1:t_fixed, 300), :, 1)
timesteps = reshape(repeat([t_fixed], size(dm_tmp100)[1]), :, 1)

#alphas[455]
#alphas[500]
Q = log10.(q_sample(dm_tmp100[:,most_var], timesteps, alphas).+1)
Q = log10.(q_sample(dm_tmp100, timesteps, alphas).+1)
Q = log10.(q_sample_binom(dm_tmp100[:,most_var], alphas[timesteps], alphas).+1)

Q = q_sample(dm_tmp100, timesteps, alphas) 
Q = q_sample(dm_tmp100[:,most_var], timesteps, alphas) 
Q = log10.(Q*(1/alphas[455]) .+ 1)
Q = log10.(q_sample(dm_tmp100, timesteps, alphas) .+1 )
Q = log10.(q_sample(dm_tmp100[:, most_var], timesteps, alphas) .+1 )

Q

Q_train = Q[train_idx, :]
Q_test =  Q[test_idx,:]
#q_sample_binom(data_matrix_100, 0.01, alphas)

dm_tmp100_log

#result = trained_model(Q', timestep_embedding(timesteps, 500)')' 
result_train = trained_model(Q_train', timesteps_train')' 
result_train = trained_model(Q_train')'
result_test = trained_model(Q_test', timesteps_test')' 
result_test = trained_model(Q_test')' 
result= cpu(trained_model(gpu(Q'), timesteps')')
result= cpu(trained_model(gpu(var_make(Q')), timesteps')')
result= trained_model(Q')'
result= cpu(trained_model(gpu(Q')))'
result= cpu(trained_model(gpu(Q_train')))'
result= cpu(trained_model(gpu(Q_test')))'

var_make(dm_tmp100_log)
mean(dm_tmp100_log,dims=1)
var_make(Q')
maximum(dm_tmp100_log)
maximum(Q)
maximum(result)
minimum(result)

dm_tmp100_log
result

CairoMakie.heatmap(cor(dm_tmp100_log[sort_idx, :],dims=2))
CairoMakie.heatmap(cor(Q[sort_idx,:],dims=2))
CairoMakie.heatmap(cor(result[sort_idx,:],dims=2))
result

train_x  = dm_tmp100[train_idx, :]
train_x_log  = log10.(train_x.+1)
timesteps_train = reshape(repeat([t_fixed], 240), :, 1)
#Q_train = q_sample_binom(train_x, alphas[timesteps_train], alphas)
Q_train = log10.(q_sample_binom(train_x, alphas[timesteps_train], alphas) .+1 )

timesteps = reshape(repeat([t_fixed], 300), :, 1)
#Q_train = q_sample_binom(train_x, alphas[timesteps_train], alphas)
Q = log10.(q_sample_binom(dm_tmp100, alphas[timesteps], alphas) .+1 )

loss(trained_model, train_x_log, Q_train, timesteps_train)


test_x  = dm_tmp100[test_idx, :]
test_x_log  = log10.(test_x.+1)
timesteps_test = reshape(repeat([t_fixed], 60), :, 1)
#Q_test = q_sample_binom(test_x, alphas[timesteps_test], alphas)
Q_test = log10.(q_sample_binom(test_x, alphas[timesteps_test], alphas) .+1 )

loss(trained_model, test_x_log, Q_test, timesteps_test)

#loss(trained_model, test_x_log, result_test, timesteps_test)


cor(vec(dm_tmp100_log[:,most_var]), vec(Q))
cor(vec(dm_tmp100_log[:,most_var]), vec(result))

f=Figure(fontsize=22)
ax, hm = CairoMakie.hexbin(f[1,1], vec(dm_tmp100_log[:,:]), vec(Q),  
                    colorscale=log10,cellsize=0.1, 
                    axis=(#title="Before denoising",
                    limits=(-0.5,8,-0.5,8),
                    xlabel="100% sequencing depth",
                    ylabel="1% sequencing depth simulated\nthrough binomial thinning"))

ax, hm = CairoMakie.hexbin(f[1,1], vec(dm_tmp100_log[:,:]), vec(result[:,:]),
                    colorscale=log10,cellsize=0.1, 
                    axis=(#title="After denoising",
                    #limits=(nothing,8,nothing,8),
                    limits=(-0.5,8,-0.5,8),
                    xlabel="100% sequencing depth",
                    ylabel="Denoised data"),
                    )

CairoMakie.ablines!( 0,1, color=[:red], "t=0", linestyle=:dash)
#CairoMakie.ylims!(ax, (-1, 11.9))
CairoMakie.Colorbar(f[1,2], hm,
    label = "Number of gene expressions",
    height = Relative(0.5)
)
f

save()


log10(1e-5)
maximum(Q)

Q
Q_o = Q
Q_v = Q .+5
Q_v = Q
Q_v[Q .<= log10(2)] .= 0
Q = Q_o
Q_o
Q[Q .<= log10(2)] .= 0
Q
Q
dm_tmp100_log[:,most_var]
result
dm_tmp100_log[:,most_var]

CairoMakie.heatmap(cor(Q_o[sort_idx,:], dims=2))

full_vec
all_vals[end]
filt = dm_tmp100[:,sortperm(var(dm_tmp100_log, dims=1)[1,:],rev=true)[1:10000]]
filt_log = log10.(filt.+1)
q_filt = log10.(q_sample_binom(filt, 0.01, alphas).+1)
res_filt = trained_model(q_filt')'

cors = cor(filt_log,dims=2)
cors = cor(q_filt,dims=2)
cors = cor(res_filt,dims=2)

cors = cor(dm_tmp100_log[sort_idx,:],dims=2)
#cors = cor(Q,dims=2)
cors = cor(Q[sort_idx,:],dims=2)
cors = cor(result[sort_idx,:],dims=2)
result
tim = reshape(repeat([455], 300), :, 1)
cors = cor(log10.(q_sample_binom(dm_tmp100, alphas[tim], alphas) .+1 ),dims=2)

CairoMakie.heatmap(cors)
minimum(cors)
CairoMakie

T=1000
T = 559
T = 455
samples = collect(1:5:size(dm_tmp100)[1])
Tim = repeat([T], length(samples))
z_t = log10.(q_sample(dm_tmp100[samples,most_var], Tim, alphas) .+1 )
Q = log10.(q_sample(dm_tmp100[samples,:], Tim, alphas) .+1 )
z_t = log10.(q_sample(dm_tmp100[samples,:], Tim, alphas) .+1 )
z_t_1 = log10.(q_sample_binom(dm_tmp100, alphas[455], alphas) .+1 ) 
z_t_1 = log10.(q_sample_binom(dm_tmp100, alphas[455], alphas) .+1 ) 

ori = z_t

maximum(dm_tmp100_log[samples,most_var])
maximum(z_t)

all_vals  = []
Z_t = z_t
push!(all_vals, vec(Z_t))

i=455
p_t = (alphas[i-1] - alphas[i])/(1-alphas[i])


Tim = repeat([i], 300)
x_hat = trained_model(z_t', Tim')'

CairoMakie.hexbin(vec(dm_tmp100_log[samples,most_var]), vec(all_vals[end]), colorscale=log10,cellsize=0.1)

deltas

Plots.plot([alphas[i-1] - alphas[i] for i in 2:559])
Plots.plot([alphas[i] for i in 1:559])

Z_t = z_t
z_t


for (frame, i) in enumerate(collect(T:-1:2))
    println(i)
    #x_hat = trained_model(z_t', timestep_embedding(repeat([i], 100), 500)')'
    Tim = repeat([i], length(samples))
    
    x_hat = cpu(trained_model(gpu(z_t'), Tim')')
    #println(maximum(x_hat))
    #rate = (alphas[i-1] - alphas[i])  # * diffusion.lbd
    #println(rate)
    #println(maximum(x_hat))
    #println()
    #z_t = [i[1] for i in rand.(Distributions.Poisson.(rate .* x_hat), 1)]./lmb
    adj_z_t = clamp.(10 .^ (z_t) .- 1, 0, Inf)
    adj_x_hat = clamp.(10 .^ (x_hat) .- 1, 0, Inf)
    
    #p_t = (alphas[i-1] - alphas[i])/(1-alphas[i]) 
    p_t = (alphas[i-1] - alphas[i])
    
    rate = adj_x_hat.*p_t
    #println(p_t)
    #z_t = log10.(adj_z_t .+ [i[1] for i in rand.(Distributions.Poisson.(rate), 1)].+1)
    z_t = log10.(adj_z_t + [i[1] for i in rand.(Distributions.Poisson.(rate), 1)].+1)



    #TODO:  timesteps.-1
    println(maximum(z_t))
    println()
    #if maximum(z_t) > curr_max+1
    #    break
    #end
    #push!(all_vals, vec(z_t))
end

for j in ProgressBar(1:Int32(8337/3-1))
    z_t = Z_t[(j-1)*3+1:j*3, :]
    for (frame, i) in enumerate(collect(T:-1:2))
    #println(i)
    #x_hat = trained_model(z_t', timestep_embedding(repeat([i], 100), 500)')'
    Tim = repeat([i], length(samples))
    
    x_hat = cpu(trained_model(gpu(z_t'), Tim[(j-1)*3+1:j*3, :]')')
    #println(maximum(x_hat))
    #rate = (alphas[i-1] - alphas[i])  # * diffusion.lbd
    #println(rate)
    #println(maximum(x_hat))
    #println()
    #z_t = [i[1] for i in rand.(Distributions.Poisson.(rate .* x_hat), 1)]./lmb
    adj_z_t = clamp.(10 .^ (z_t) .- 1, 0, Inf)
    adj_x_hat = clamp.(10 .^ (x_hat) .- 1, 0, Inf)
    
    #p_t = (alphas[i-1] - alphas[i])/(1-alphas[i]) 
    p_t = (alphas[i-1] - alphas[i])
    
    rate = adj_x_hat.*p_t
    #println(p_t)
    #z_t = log10.(adj_z_t .+ [i[1] for i in rand.(Distributions.Poisson.(rate), 1)].+1)
    z_t = log10.(adj_z_t + [i[1] for i in rand.(Distributions.Poisson.(rate), 1)].+1)



    #TODO:  timesteps.-1
    #println(maximum(z_t))
    #println()
    #if maximum(z_t) > curr_max+1
    #    break
    #end
    
end
push!(all_vals, vec(z_t))
end
push!(all_vals, vec(z_t))

std(vec(cor(z_t_end, dims=2)))
CairoMakie.heatmap(cor(z_t_end, dims=2))
CairoMakie.heatmap(cor(dm_tmp100_log[samples,:], dims=2))

# More direct : how much it corresponds to its nearest neighbor.
# Before solving, put in place the setup to prove it does what I want
# Eval or make it until I feel it works?

# Convergence time as a function of the dataset -> Could maybe ibe interesting
# To anticipate further conclusions

vec(dm_tmp100_log[samples,:])
corel = Observable(cor(vec(dm_tmp100_log[samples,:]), vec(Q)))
corel2 = Observable(corspearman(vec(dm_tmp100_log[samples,:]), vec(all_vals[end])))
points = Observable(vec(Q))
iter = Observable(1)
points_mat = reshape(all_vals[end], 834,19962)

dm_tmp100_log

## ; Spearman: $(@sprintf("%.4f", corel2[]))
f,ax, hm = CairoMakie.hexbin(vec(dm_tmp100_log[samples,:]), points, colorscale=log10,cellsize=0.1, 
        aspect = AxisAspect(1), figure=(;fontsize=20), 
        axis=(ylabel="Denoised log-counts",
            xlabel="20M reads log-counts", limits=(-0.5,8,-0.5,8),
            title="Timestep: $(@sprintf("%03d", iter[]))\nPearson: $(@sprintf("%.4f", corel[]))"))
#ax, hm = CairoMakie.hexbin(f[1,1], test_res, full_test,  colorscale=log10,cellsize=0.2, 
#        aspect = AxisAspect(1), axis=(ylabel="predicted log-counts",xlabel="original log-counts", title="Prediction after complete \ndiffusion denoising process"))

#Axis(f[1,1], xlabel="test")
CairoMakie.ablines!( 0,1, color=[:red], "t=0", linestyle=:dash)
#CairoMakie.ylims!(ax, (0,11))
#ax

#CairoMakie.scatter!(ax, dm_vec, result_vec)
CairoMakie.Colorbar(f[1,2], hm,
    label = "Number of expressions",
    height = Relative(0.5)
)
f

f
save("noised_hb_200k.svg",f)

all_vals

frames = 1:length(all_vals)

record(f, "allvar_diff_10epoch.mp4", frames;
        framerate = 30) do frame
    ax.title="Iteration: $(@sprintf("%03d", frame)); Pearson: $(@sprintf("%.4f", cor(vec(dm_tmp100_log[samples,:]), vec(all_vals[frame]))))"
    points[] = all_vals[frame]
    corel[] = cor(vec(dm_tmp100_log[samples,:]), vec(all_vals[frame]))
    iter[] = frame
end

sample_types = types_z[samples]
order=sortperm(sample_types)
sample_types=sample_types[order]
cors=cor(z_t[order,:], dims=2)
cors=cor(Q[order,:], dims=2)
cors=cor(dm_tmp100_log[samples,:][order,:], dims=2)


s_z_t=z_t[order,:]
s_Q=Q[order,:]
s_dm = dm_tmp100_log[samples,:][order,:]

z_t_tmp = z_t

f,ax,hm = CairoMakie.heatmap(cors)
CairoMakie.Colorbar(f[1,2], hm,
    label = "Correlation",
    height = Relative(0.5)
)
f