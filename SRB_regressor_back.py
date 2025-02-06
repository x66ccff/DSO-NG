
import sympy as sp


from dso import DeepSymbolicRegressor



est = DeepSymbolicRegressor("/home/ruankai/_Project/uDSRPSRN/deep-symbolic-optimization/conf_SRB.json")


def complexity(est):
    expr = est.program_.sympy_expr
    return sp.count_ops(expr)


def model(est):
    expr = str(est.program_.sympy_expr)
    return expr



hyper_params = [{}]

eval_kwargs = {
    "test_params": dict(
        device='cuda',
    )
}
