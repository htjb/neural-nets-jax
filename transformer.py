import jax
import time
from jax import numpy as jnp
from jax import random
from jax import jit

# create a set of random values to embed (e.g. positions of objects we are
# attempting to find relationships between)

key = random.PRNGKey(0)
split_key, key = random.split(key)

nbatches = 10
batch_size = 3
positions = random.uniform(split_key, (nbatches, batch_size, 2), 
                           minval=-5, maxval=5)


def initialise_embedding_layer(size, key, scale=1e-2):

    w_key, b_key = random.split(key)
    return [scale*random.normal(w_key, (size[0], size[1])),
            scale*random.normal(b_key, (size[1],))]


@jit
def linear_layer(params, input):
    return jnp.dot(input, params[0]) + params[1]


def initialise_transformer(embedded_dim, key, scale=1e-2):

    w_query_key, key = random.split(key)
    w_key_key, w_value_key = random.split(key)
    return {'query': scale*random.normal(w_query_key, (embedded_dim, embedded_dim)),
            'key': scale*random.normal(w_key_key, (embedded_dim, embedded_dim)),
            'value': scale*random.normal(w_value_key, (embedded_dim, embedded_dim))}

@jit
def transformer(params, input):
    query = jnp.dot(input, params['query'])
    key = jnp.dot(input, params['key'])
    
    query_key = jnp.dot(key.T, query)
    softmax = jnp.exp(query_key) / jnp.sum(jnp.exp(query_key))

    value = jnp.dot(input, params['value'])

    output = jnp.dot(value, softmax)

    return output


embedding_dim = 64
split_key, key = random.split(key)
params = initialise_embedding_layer([2, embedding_dim], split_key)


# for some reason linear layer can handle batches without me explicitly batching it
#batched_linear_layer = jax.vmap(linear_layer, in_axes=(None, 0))
embedded_positions = linear_layer(params, positions)

split_key, key = random.split(key)
transformer_params = initialise_transformer(embedding_dim, split_key)
batched_transformer = jax.vmap(transformer, in_axes=(None, 0))
out = batched_transformer(transformer_params, embedded_positions)

print(out.shape)
