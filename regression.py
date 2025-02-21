import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from jax import random
import jax

key = random.PRNGKey(0)

# generate training/test/validation data 
n = 1000
x = jnp.linspace(-10, 10, 100)
keys = random.split(key, num=3)
print(keys)
key = keys[2]
mu = random.uniform(keys[0], shape=(n,), minval=-3, maxval=3)
std = random.uniform(keys[1], shape=(n,),  minval=0.1, maxval=2)
parameters = jnp.stack([mu, std])

@jax.jit
def gaussian(x, params):
    return jnp.exp(-0.5 * ((x - params[0]) / params[1]) ** 2)


signals = jax.vmap(gaussian, in_axes=(None, 1))(x, parameters)

train_signals, test_signals, train_parameters, test_parameters = \
    train_test_split(signals, parameters.T, test_size=0.3)

test_signals, val_signals, test_parameters, val_parameters = \
    train_test_split(test_signals, test_parameters, test_size=0.5)

layers = [2, 20, 20, 100]


def initialise(layers, key, scale=1e-2):

    def layer(inn, out, ks, scale):
        return {'weights': scale*random.normal(ks[0], (inn, out)),
                'bias': scale*random.normal(ks[1], (out,))}

    kks = random.split(key, num=len(layers)*2)
    kks = kks.reshape(len(layers), 2, 2)

    ll = {}
    for i in range(len(layers)-1):
        ll['layer' + str(i)] = layer(layers[:-1][i], 
                                     layers[1:][i], kks[i], scale)
    return ll


params = initialise(layers, key)

def forward(params, input):

    @jax.jit
    def layer_pass(pp, input):
        return jnp.dot(input, pp['weights']) + pp['bias']
    
    output = layer_pass(params['layer0'], input)
    for i in range(len(params)-1):
        output = layer_pass(params['layer' + str(i+1)], output)
    return output

@jax.jit
def lossf(pred, truth):
    return jnp.mean(jnp.square(pred-truth))


pred = forward(params, train_parameters)
loss = lossf(pred, train_signals)


plt.plot(x, pred[0])
plt.plot(x, train_signals[0])
plt.show()