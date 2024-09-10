from diffrax import diffeqsolve, Dopri5, ODETerm, PIDController, BacksolveAdjoint
from func import IntensityODEFunc
from jaxtyping import Float, Array, PyTree

def integrate(func: IntensityODEFunc, t0: Float[Array, "1"], t1: Float[Array, "1"], x0: PyTree, args: PyTree):
    solution = diffeqsolve(
        ODETerm(func), 
        Dopri5(),
        t0,
        t1,
        None,
        x0,
        args,
        stepsize_controller=PIDController(rtol=1e-4, atol=1e-4),
        adjoint=BacksolveAdjoint(),
        max_steps=2 ** 31 - 1
    )
    x1 = [y[-1] for y in solution.ys]
    return x1
