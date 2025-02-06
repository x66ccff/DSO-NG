from .base import TokenGenerator
import re
import numpy as np
from pandas import read_csv
import random
import yaml
import json
import itertools
from .GP.model.config import Config
from .GP.model.pipeline import Pipeline
import sympy as sp
import random
import itertools

# CONST_LIST = ['-1','1','pi'] # SRbench
MAX_LEN_SET = 1000
SAMPLE_PROB = 0.5
SAMPLE_PROB_CROSS_VAR = 0.5
MAX_INTEGER = 10


def read_yaml_to_json(file_path):
    with open(file_path, "r") as file:
        try:
            config = yaml.safe_load(file)
            return json.dumps(config)
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None


def get_max_depth(expr):
    def traverse(subexpr, current_depth):
        max_depth = current_depth
        for arg in subexpr.args:
            max_depth = max(max_depth, traverse(arg, current_depth + 1))
        return max_depth

    return traverse(expr, 0)


def get_subexpressions_at_depth(expr, depth):
    if depth == 0:
        return [expr]

    subexprs = []

    def traverse(subexpr, current_depth):
        if current_depth == depth:
            subexprs.append(subexpr)
        else:
            for arg in subexpr.args:
                traverse(arg, current_depth + 1)

    traverse(expr, 0)
    return subexprs


def get_last_subexprs(expr, depth=4):
    max_depth = get_max_depth(expr)
    # print(f"Maximum depth of expr1: {max_depth}")

    ret = []
    for d in range(0, max_depth + 1):
        subexprs = get_subexpressions_at_depth(expr, d)
        # print(f"Depth {d}: {subexprs}")
        if max_depth - d < depth:
            ret.extend(subexprs)
    ret = list(set(ret))
    ret.sort(key=lambda x: x.count_ops())
    return ret


def has_large_integer(expr):
    if isinstance(expr, str):
        expr = sp.S(expr)
    for atom in expr.atoms():
        if isinstance(atom, sp.Integer) and abs(int(atom)) > MAX_INTEGER:
            return True
        if isinstance(atom, sp.Rational) and (
            abs(atom.p) > MAX_INTEGER or abs(atom.q) > MAX_INTEGER
        ):
            return True

    return False


def generate_cross_variable(variables, n_sample):
    operations = ["*", "+", "/"]
    all_combinations = list(itertools.combinations_with_replacement(variables, 2))
    if len(all_combinations) < n_sample:
        n_sample = len(all_combinations)

    sampled_combinations = random.sample(all_combinations, n_sample)
    cross_variables = []

    for var1, var2 in sampled_combinations:
        op = random.choice(operations)
        if var1 == var2:
            if op == "/":
                cross_variables.append(f"{var1}")
            else:
                cross_variables.append(f"{var1}{op}{var1}")
        elif op == "/" and random.random() < 0.5:
            cross_variables.append(f"{var2}{op}{var1}")
        else:
            cross_variables.append(f"{var1}{op}{var2}")

    # print('cross_variables:',cross_variables)
    return cross_variables


class GP_TokenGenerator(TokenGenerator):
    def __init__(
        self,
        regressor,
        config,
        variables,
        operators_op,
        use_const,
        n_inputs,
        use_extra_const=False,
    ):

        self.config = config
        self.variables = variables
        self.operators = operators_op
        self.use_const = use_const
        self.n_inputs = n_inputs
        self.token = []
        self.token_generator_model = None
        self.visited_set = set()

        self.trying_const_range = regressor.trying_const_range

        self.CONST_LB = self.trying_const_range[0]
        self.CONST_UB = self.trying_const_range[1]

        self.use_extra_const = use_extra_const
        self.EXTRA_CONST = ["pi"]

    def sample_const(self, use_float_const):
        if use_float_const:
            return round(random.uniform(self.CONST_LB, self.CONST_UB), 1)
        else:
            CONST_LIST = np.linspace(
                self.CONST_LB, self.CONST_UB, self.CONST_UB - self.CONST_LB + 1
            ).tolist()
            CONST_LIST = [x for x in CONST_LIST if x != 0]
            if self.use_extra_const:
                CONST_LIST.extend(self.EXTRA_CONST)
            return random.choice(CONST_LIST)

    def step(
        self,
        n_psrn_tokens,
        n_sample_variables,
        X,
        y,
        use_set=True,
        reset=False,
        use_float_const=False,
    ):

        n_tokens = n_psrn_tokens - n_sample_variables

        y = y.reshape(-1)
        X = np.transpose(X, (1, 0))
        print("X.shape", X.shape)
        print("y.shape", y.shape)

        best_expr, all_expr_form = self.token_generator_fit_x_y(
            X, y, self.config, reset=reset
        )
        print("all_expr_form", all_expr_form)
        symbols = self.process_all_form_to_tokens(all_expr_form, use_float_const)

        best_expr = self.replace_varname([best_expr])[0]

        print("subexpress")
        symbols_sympy = [sp.S(str(sym)) for sym in symbols]
        tokens = []

        symbols_sympy += (
            [e.expand() for e in symbols_sympy]
            + [e.together() for e in symbols_sympy]
            + [e.powsimp() for e in symbols_sympy]
            + [e.radsimp() for e in symbols_sympy]
        )

        for expr in symbols_sympy:
            subexpr = get_last_subexprs(expr)
            for e in subexpr:
                if e.count_ops() < 10:
                    tokens.append(e)
        tokens = list(set(tokens))

        from collections import Counter

        token_counts = Counter(tokens)
        all_tokens = list(token_counts.keys())
        frequencies = list(token_counts.values())
        tokens_freq = list(zip(all_tokens, frequencies))

        n_try = 0
        keep_try = True
        while keep_try:

            n_try += 1
            if n_try > MAX_LEN_SET:
                keep_try = False
            if len(all_tokens) > n_tokens:
                sampled_tokens = []
                cnt_cross_variables = 0

                n_try_2 = 0
                while len(sampled_tokens) < n_tokens:
                    n_try_2 += 1
                    if n_try_2 > MAX_LEN_SET:
                        sampled_tokens = random.choices(
                            all_tokens, weights=frequencies, k=n_tokens
                        )
                        break
                    if random.random() < SAMPLE_PROB:
                        if random.random() < SAMPLE_PROB_CROSS_VAR:
                            chosen_token = sp.S(
                                generate_cross_variable(self.variables, 1)[0]
                            )
                        else:
                            chosen_token = sp.S(self.sample_const(use_float_const))
                    else:
                        chosen_token = sp.S(
                            random.choices(
                                [token for token, freq in tokens_freq],
                                weights=[freq for token, freq in tokens_freq],
                                k=1,
                            )[0]
                        )
                    from utils.exprutils import has_nested_func

                    if chosen_token is None:
                        continue
                    if (
                        not (not use_float_const and "." in str(chosen_token))
                        and str(chosen_token) not in self.variables
                        and not has_nested_func(chosen_token)
                        and not has_large_integer(chosen_token)
                    ):
                        if chosen_token not in sampled_tokens:
                            sampled_tokens.append(chosen_token)
                        else:
                            # print('#!', end='')
                            pass
                    else:
                        # print('#!', end='')
                        pass
            else:
                sampled_constants_num = n_tokens - len(tokens)
                sampled_constants = [
                    self.sample_const(use_float_const)
                    for i in range(sampled_constants_num)
                ]
                # print('sampled_constants',sampled_constants)
                sampled_tokens = tokens + sampled_constants

            # print('sampled_tokens',sampled_tokens)
            sampled_tokens = [str(t) for t in sampled_tokens]

            if use_set:
                sampled_set = set(sampled_tokens)
                if len(sampled_set) != len(set(sampled_tokens + self.variables)) - len(
                    self.variables
                ):
                    continue

                if str(sampled_set) not in self.visited_set:
                    self.visited_set.add(str(sampled_set))
                    return best_expr, sampled_tokens
                else:
                    continue
            else:
                return best_expr, sampled_tokens

        return best_expr, sampled_tokens

    def process_all_form_to_tokens(self, all_expr_form, use_float_const):
        new_expr_forms = []
        for expr in all_expr_form:
            if len(expr) == 0:
                continue
            elif expr.endswith("+"):
                expr = expr[:-1]
            elif expr.endswith("**"):
                expr = expr[:-2]
            elif expr.endswith("*"):
                expr = expr[:-1]
            expr = expr.replace("C", str(self.sample_const(use_float_const)))
            new_expr_forms.append(expr)

        new_expr_forms = list(set(new_expr_forms))
        new_expr_forms = self.replace_varname(new_expr_forms)
        return new_expr_forms

    def replace_varname(self, new_expr_forms):
        for i in range(len(new_expr_forms)):
            for j in range(len(self.variables)):
                new_expr_forms[i] = new_expr_forms[i].replace(
                    "X" + str(j + 1), self.variables[j]
                )
        return new_expr_forms

    def reward(self, reward, expressions):

        self.token_generator_model.use_psrn_reward_expressions_to_update(expressions)

    def token_generator_fit_x_y(self, x, y, gp_config, reset=False):
        # x.shape (2, MAX_LEN_SET) y.shape (MAX_LEN_SET,)

        if self.token_generator_model is None or reset:
            config = Config()
            config.json(gp_config)
            config.set_input(x=x, t=y, x_=x, t_=y, tokens=gp_config["base"]["tokens"])
            print("=" * 40)
            print("GP config:", config)
            print("=" * 40)
            self.token_generator_model = Pipeline(config=config)
            clear = True
        else:
            clear = False

        best_exprs, exprs = self.token_generator_model.fit(clear=clear)
        exprs = [best_exprs] + exprs
        return best_exprs, exprs
