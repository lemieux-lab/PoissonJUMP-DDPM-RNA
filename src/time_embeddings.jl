function timestep_embedding(timesteps, embed_dim::Int)
    half_dim = embed_dim // 2
    embed = log(10000) / (half_dim - 1)
    embed = exp.((.-collect(0:half_dim-1)) .* embed)
    #println(embed)
    #println(size(embed))
    #println(size(timesteps))
    #println(size(embed))
    #println(size(timesteps))
    embed = timesteps' .* embed'
    #println(embed)

    embed = hcat(sin.(embed), cos.(embed))
    
    if embed_dim % 2 == 1
        embed = pad_constant(embed, (0,1), 0)  # padding the last dimension
    end
    return embed
end



