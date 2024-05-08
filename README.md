<h1 align="center">typst_pyimage</h1>

<p align="center">Wraps <a href="https://github.com/typst/typst">Typst</a> to support inline Python code for generating content and figures.</p>

## Example

<img align="right" width="45%" src="https://raw.githubusercontent.com/patrick-kidger/typst_pyimage/main/imgs/lotka_volterra.png">

```typst
#import ".typst_pyimage/pyimage.typ": pyimage

Consider the Lotka--Volterra (predator-prey)
equations:

#pyimage("
import diffrax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def func(t, y, args):
  rabbits, cats = y
  d_rabbits = rabbits - rabbits*cats
  d_cats = -cats + rabbits*cats
  return d_rabbits, d_cats

term = diffrax.ODETerm(func)
solver = diffrax.Tsit5()
y0 = (2, 1)
t0 = 0
t1 = 20
dt0 = 0.01
ts = jnp.linspace(t0, t1, 100)
saveat = diffrax.SaveAt(ts=ts)
sol = diffrax.diffeqsolve(term, solver, t0,
                          t1, dt0, y0,
                          saveat=saveat)

plt.plot(ts, sol.ys[0], label='Rabbits')
plt.plot(ts, sol.ys[1], label='Cats')
plt.xlim(0, 20)
plt.ylim(0, 2.5)
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
", width: 70%)
```

_(This example uses [JAX](https://github.com/google/jax) and [Diffrax](https://github.com/patrick-kidger/diffrax) to solve an ODE.)_

## Installation

```
pip install typst_pyimage
```

This requires that you're using Typst locally -- it won't work with the web app.

## Usage

1. Import `pyimage.typ`. At the start of your `.typ` file, add the line `#import ".typst_pyimage/pyimage.typ": pyimage, pycontent, pyinit`.

2. Use these functions.

   a. `pyimage(string, ..arguments) -> content`. The positional string should be a Python program that creates a single matplotlib figure. Any named arguments are forwarded on to Typst's built-in `image` function. You can use it just like the normal `image` function, e.g. `#align(center, pyimage("..."))`.

   b. `pycontent(string)`. The positional string should be a Python program that produces a string on its final line. This string will be treated as Typst code.

   c. `pyinit(string)`. The positional string should be a Python program. This will be evaluated before all `pyimage` or `pycontent` calls, e.g. to perform imports or other setup.

3. Compile or watch. Run either of the following two commands:

   ```
   python -m typst_pyimage compile your_file.typ
   python -m typst_pyimage watch your_file.typ
   ```

   This will extract and run all your Python code. In addition it will call either `typst compile your_file.typ` or `typst watch your_file.typ`.

   The resulting images are saved in the `.typst_pyimage` folder.

   For more information on the available arguments, run `python -m typst_pyimage -h`.

## Notes

It's common to have an initial block of code that is in common to all `#pyimage("...")` and `#pycontent("...")` calls (such as import statements, defining helpers etc). These can be placed in a `#pyinit("...")` directive.

Each `#pyimage("...")` block is executed as a fresh module (i.e. as if each was a separate Python file), but with the same Python interpreter.

Overall, this is essentially equivalent to the following Python code:

```
# main.py
import pyinit
import pyimage1
import pyimage2

# pyinit.py
...  # your #pyinit("...") code

# pyimage1.py
from pyinit import *
...  # your first #pyimage("...") code

# pyimage2.py
from pyinit import *
...  # your second #pyimage("...") code
```

This means that e.g. any global caches will be shared across all `#pyimage("...")` calls. (Useful when using a library like JAX, which has a JIT compilation cache.)

## Limitations

1. The watcher just extracts all the `pyimage("...")` etc. blocks via regex, and runs them in the order that they appear in the file. This means that (a) the `"` character may not appear anywhere in the Python code (even if escaped), and (b) trying to call `pyimage` etc. dynamically (i.e. not with a literal string at the top level of your program) will not work.
2. Only `pyimage("...")` etc. calls inside the single watched file are tracked.

We could probably lift 1a and 2 with a bit of effort. PRs welcome.
