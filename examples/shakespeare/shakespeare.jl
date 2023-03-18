include("imports.jl")

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
head_size = n_embd ÷ n_head
n_layer = 6
dropout = 0.2

Random.seed!(1234)
device = CUDA.functional() ? gpu : cpu

struct GPTLanguageModel
    token_embedding_table
    position_embedding_table
    blocks
    layer_norm
    lm_head
end

function GPTLanguageModel(vocab_size, block_size, n_embd)
    GPTLanguageModel(
        # each token directly reads off the logits for the next token from a lookup table
        Embedding(vocab_size=>n_embd),
        Embedding(block_size=>n_embd),
        BlockList([Block(n_embd; num_heads=n_head, dropout=dropout) for _ in 1:n_layer]),
        LayerNorm(n_embd), # final layer norm
        Dense(n_embd, vocab_size),
    )
end

Functors.@functor GPTLanguageModel

(m::GPTLanguageModel)(idx; mask=nothing) = m(idx, nothing; mask=mask)

function (m::GPTLanguageModel)(idx, targets; mask=nothing)
    T, B = size(idx)
    tok_emb = m.token_embedding_table(idx) # (C,T,B)
    pos_emb = m.position_embedding_table(device(1:T)) # (C,T)
    emb = tok_emb .+ pos_emb # (C,T,B)
    x = m.blocks(emb; mask=mask) # (C,T,B)
    x2 = m.layer_norm(x) # (C,T,B)
    logits = m.lm_head(x2) # (vocab_size,T,B)
    if isnothing(targets)
        loss = nothing
    else
        C, B, T = size(logits)
        logits_reshaped = reshape(logits, C, T*B)
        targets_reshaped = reshape(targets, T*B)
        targets_onehot = Flux.onehotbatch(targets_reshaped, 1:vocab_size)
        loss = Flux.logitcrossentropy(logits_reshaped, targets_onehot)
    end
    (logits=logits, loss=loss)
end

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
text = read("input.txt", String)

# here are all the unique characters that occur in this text
chars = (sort ∘ collect ∘ Set)(text)
vocab_size = length(chars)

# create a mapping from characters to integers
stoi = Dict([(ch,i) for (i,ch) in enumerate(chars)])
itos = Dict([(i,ch) for (i,ch) in enumerate(chars)])

# encoder: take a string, output a list of integers
encode(s) = [stoi[c] for c in s]

# decoder: take a list of integers, output a string
decode(l) = join([itos[i] for i in cpu(l)], "")

# Train and test splits
data = encode(text)
n = Int(round(0.9*length(data))) # first 90% will be train, rest val
train_data = data[1:n]
val_data = data[n:end]

# data loading
function getbatch(split)
    # generate a small batch of data of inputs x and targets y
    data = split == "train" ? train_data : val_data
    ix = rand(1:(length(data) - block_size), batch_size)
    x = reduce(hcat, [data[i:i+block_size-1] for i in ix])
    y = reduce(hcat, [data[i+1:i+block_size] for i in ix])
    x, y = device(x), device(y)
    x, y
end

function estimateloss(model, mask)
    out = Dict()
    testmode!(model)
    for split in ["train", "val"]
        losses = zeros(eval_iters)
        for k in 1:eval_iters
            X, Y = getbatch(split)
            logits, loss = model(X,Y; mask=mask)
            losses[k] = loss
        end
        out[split] = mean(losses)
    end
    trainmode!(model)
    out
end

function printsample(model)
    println("################### SAMPLE ###################")
    println(gensample(model))
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
end

function gensample(model)
    idx = generate(model, ones(Int, 1, 1), 1000)
    decode(idx)
end

function generate(model::GPTLanguageModel, context, max_new_tokens)
    m = cpu(model)
    testmode!(m)
    idx = context |> cpu
    # idx is (B, T) array of indices in the current context
    for _ in 1:max_new_tokens
        # crop idx to the last block_size tokens
        idx_cond = @view idx[max(end-block_size+1,1):end, :]
        # get the predictions
        logits, _ = m(idx_cond)
        # focus only on the last time step
        logits = logits[:, :, 1] # becomes (B, C)
        # apply softmax to get probabilities
        probs = Flux.softmax(logits, dims=1) # (B, C)
        # sample from the distribution
        id_next = Distributions.Categorical(probs[:,end]) |> rand
        # append sampled index to the running sequence
        idx = vcat(idx, [id_next])
        print(decode(id_next))
    end
    trainmode!(m)
    idx
end

# initialize the model
model = GPTLanguageModel(vocab_size, block_size, n_embd) |> device

# generate from the model
printsample(model)

# initialize the Flux optimizer
optim = Flux.setup(Flux.AdamW(learning_rate), model)

function train!(model)
    trainmode!(model)
    batch_mask = Float32.(tril(fill(-Inf, block_size, block_size), -1)) |> device
    @showprogress for iter in 1:max_iters
        xb, yb = getbatch("train")
        # every once in a while evaluate the loss on train and val sets
        if iter == 1 || iter % eval_interval == 0 || (iter == max_iters)
            println("\nestimating loss...")
            losses = estimateloss(model, batch_mask)
            println("step $(iter): train loss $(round(losses["train"], digits=4)), val loss $(round(losses["val"], digits=4))")
            printsample(model)
        end
        loss, grads = Flux.withgradient(model) do m
            m(xb, yb; mask=batch_mask).loss
        end
        Flux.update!(optim, model, grads[1])
    end
    testmode!(model)
end

# train the model
train!(model)

# generate from the model
printsample(model)