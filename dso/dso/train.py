"""Defines main training loop for deep symbolic optimization."""

import os
import json
import time
import itertools  # 
from itertools import compress


import sympy as sp
import torch
import random


# 主要改动点：
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()  # 禁用v2行为

import numpy as np
import pandas as pd

from symengine import sympify as se_sympify

from dso.program import Program, from_tokens
from dso.utils import empirical_entropy, get_duration, weighted_quantile, pad_action_obs_priors
from dso.memory import Batch, make_queue
from dso.variance import quantile_variance

# Ignore TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set TensorFlow seed
tf.set_random_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

import torch
# from model.regressor import PSRN_Regressor
from dso.PTS.model.regressor import PSRN_Regressor
from dso.PTS.utils.data import expr_to_Y_pred
from dso.PTS.model.models import PSRN

os.environ["CUDA_VISIBLE_DEVICES"] = str(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

operators = ['Add','Mul','SemiSub','SemiDiv','Identity','Sin','Cos','Exp','Log']

print(operators)
# operators = eval(operators)
# print(operators)

n_down_sample = 20
n_inputs = 5

topk = 20
seed = 0
experiment_name = "_"

use_constant = False

hp = {
    "operators": operators,
    "n_down_sample": n_down_sample,
    "n_inputs": n_inputs,
    "topk": topk,
    "seed": seed,
}

# 设置环境变量
os.environ['PYTHONHASHSEED'] = str(seed)

# 设置Python的random模块随机种子
random.seed(seed)

# 设置NumPy的随机种子
np.random.seed(seed)

# 设置PyTorch的随机种子
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


path_log = "./log/custom_data/" + experiment_name + "/"

if not os.path.exists(path_log):
    os.makedirs(path_log)

cnt_success = 0
sum_time = 0

psrn_model = PSRN(n_variables=n_inputs, operators=operators, n_symbol_layers=3, dr_mask=None, device="cuda")
print(psrn_model)

visited_set = set()
    
# Work for multiprocessing pool: compute reward
def work(p):
    """Compute reward and return it with optimized constants"""
    r = p.r
    return p

class Trainer():
    def __init__(self, sess, policy, policy_optimizer, gp_controller, logger,
                 pool, n_samples=2000000, batch_size=1000, alpha=0.5,
                 epsilon=0.05, verbose=True, baseline="R_e",
                 b_jumpstart=False, early_stopping=True, debug=0,
                 use_memory=False, memory_capacity=1e3,  warm_start=None, memory_threshold=None,
                 complexity="token", const_optimizer="scipy", const_params=None,  n_cores_batch=1):

        """
        Initializes the main training loop.

        Parameters
        ----------
        sess : tf.Session
            TensorFlow Session object.
        
        policy : dso.policy.Policy
            Parametrized probability distribution over discrete objects.
            Used to generate programs and compute loglikelihoods.

        policy_optimizer : dso.policy_optimizer.policy_optimizer
            policy_optimizer object used to optimize the policy.

        gp_controller : dso.gp.gp_controller.GPController or None
            GP controller object used to generate Programs.

        logger : dso.train_stats.StatsLogger
            Logger to save results with

        pool : multiprocessing.Pool or None
            Pool to parallelize reward computation. For the control task, each
            worker should have its own TensorFlow model. If None, a Pool will be
            generated if n_cores_batch > 1.

        n_samples : int or None, optional
            Total number of objects to sample. This may be exceeded depending
            on batch size.

        batch_size : int, optional
            Number of sampled expressions per iteration.

        alpha : float, optional
            Coefficient of exponentially-weighted moving average of baseline.

        epsilon : float or None, optional
            Fraction of top expressions used for training. None (or
            equivalently, 1.0) turns off risk-seeking.

        verbose : bool, optional
            Whether to print progress.

        baseline : str, optional
            Type of baseline to use: grad J = (R - b) * grad-log-prob(expression).
            Choices:
            (1) "ewma_R" : b = EWMA(<R>)
            (2) "R_e" : b = R_e
            (3) "ewma_R_e" : b = EWMA(R_e)
            (4) "combined" : b = R_e + EWMA(<R> - R_e)
            In the above, <R> is the sample average _after_ epsilon sub-sampling and
            R_e is the (1-epsilon)-quantile estimate.

        b_jumpstart : bool, optional
            Whether EWMA part of the baseline starts at the average of the first
            iteration. If False, the EWMA starts at 0.0.

        early_stopping : bool, optional
            Whether to stop early if stopping criteria is reached.

        debug : int, optional
            Debug level, also passed to Controller. 0: No debug. 1: Print initial
            parameter means. 2: Print parameter means each step.

        use_memory : bool, optional
            If True, use memory queue for reward quantile estimation.

        memory_capacity : int
            Capacity of memory queue.

        warm_start : int or None
            Number of samples to warm start the memory queue. If None, uses
            batch_size.

        memory_threshold : float or None
            If not None, run quantile variance/bias estimate experiments after
            memory weight exceeds memory_threshold.

        complexity : str, optional
            Not used

        const_optimizer : str or None, optional
            Not used

        const_params : dict, optional
            Not used

        n_cores_batch : int, optional
            Not used


        """
        self.sess = sess
        # Initialize compute graph
        self.sess.run(tf.global_variables_initializer())

        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.gp_controller = gp_controller
        self.logger = logger
        self.pool = pool
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.alpha = alpha
        self.epsilon = epsilon
        self.verbose = verbose
        self.baseline = baseline
        self.b_jumpstart = b_jumpstart
        self.early_stopping = early_stopping
        self.debug = debug
        self.use_memory = use_memory
        self.memory_threshold = memory_threshold

        if self.debug:
            tvars = tf.trainable_variables()
            def print_var_means():
                tvars_vals = self.sess.run(tvars)
                for var, val in zip(tvars, tvars_vals):
                    print(var.name, "mean:", val.mean(), "var:", val.var())
            self.print_var_means = print_var_means

        # Create the priority_queue if needed
        if hasattr(self.policy_optimizer, 'pqt_k'):
            from dso.policy_optimizer.pqt_policy_optimizer import PQTPolicyOptimizer
            assert type(self.policy_optimizer) == PQTPolicyOptimizer
            # Create the priority queue
            k = self.policy_optimizer.pqt_k
            if k is not None and k > 0:
                self.priority_queue = make_queue(priority=True, capacity=k)
        else:
            self.priority_queue = None

        # Create the memory queue
        if self.use_memory:
            assert self.epsilon is not None and self.epsilon < 1.0, \
                "Memory queue is only used with risk-seeking."
            self.memory_queue = make_queue(policy=self.policy, priority=False,
                                           capacity=int(memory_capacity))

            # Warm start the queue
            # TBD: Parallelize. Abstract sampling a Batch
            warm_start = warm_start if warm_start is not None else self.batch_size
            actions, obs, priors = policy.sample(warm_start)
            programs = [from_tokens(a) for a in actions]
            r = np.array([p.r for p in programs])
            l = np.array([len(p.traversal) for p in programs])
            on_policy = np.array([p.originally_on_policy for p in programs])
            sampled_batch = Batch(actions=actions, obs=obs, priors=priors,
                                  lengths=l, rewards=r, on_policy=on_policy)
            self.memory_queue.push_batch(sampled_batch, programs)
        else:
            self.memory_queue = None

        self.nevals = 0 # Total number of sampled expressions (from RL or GP)
        self.iteration = 0 # Iteration counter
        self.r_best = -np.inf
        self.p_r_best = None
        self.done = False

    def run_one_step(self, override=None):
        """
        Executes one step of main training loop. If override is given,
        train on that batch. Otherwise, sample the batch to train on.

        Parameters
        ----------
        override : tuple or None
            Tuple of (actions, obs, priors, programs) to train on offline
            samples instead of sampled
        """
        positional_entropy = None
        top_samples_per_batch = list()
        if self.debug >= 1:
            print("\nDEBUG: Policy parameter means:")
            self.print_var_means()

        ewma = None if self.b_jumpstart else 0.0 # EWMA portion of baseline

        start_time = time.time()
        if self.verbose:
            print("-- RUNNING ITERATIONS START -------------") # KKTODO




            # try:
            #     df_pf = self.logger.get_df_pf()
            #     print(df_pf[["expression","complexity","r","nmse_test"]])
            #     pf_expressions = df_pf["expression"].tolist()
            #     print("pf_expressions")
            #     print(pf_expressions)
                

            # except IndexError as e:
            #     print("No available df pf.")
            
            
            # test_expressions = [
            #     # 这些应该会抛出错误
            #     # "(x+1)",  # 使用了不明确的x
            #     # "0.5*x1",  # 如果library中没有const会报错

            #     # 这些应该正常工作
            #     # "(x1+x2)",
            #     "(x1*x1)",
            #     "(((x1)*(x1))+(sin(x1)))",
            #     # 如果library有const：
            #     # "(x1+1)",
            #     # "(0.5*x1)",
            # ]

            # for expr in test_expressions:
            #     print(expr)
            #     result = Program.from_bracket_string(expr)
            #     print(f"{expr} -> {result}")
            # expr_kk = Program.from_bracket_string("(((x)*(x))+(sin(x)))")
            # print("expr_kk", expr_kk)
            
            
            

        # Number of extra samples generated during attempt to get
        # batch_size new samples
        n_extra = 0
        # Record previous cache before new samples are added by from_tokens
        s_history = list(Program.cache.keys())

        # Construct the actions, obs, priors, and programs
        # Shape of actions: (batch_size, max_length)
        # Shape of obs: (batch_size, obs_dim, max_length)
        # Shape of priors: (batch_size, max_length, n_choices)
        if override is None:
            # Sample batch of Programs from the Controller
            actions, obs, priors = self.policy.sample(self.batch_size) ################## 这个应该是RNN的样本出来的地方，关键应该是这个 sample函数
            programs = [from_tokens(a) for a in actions]            
        else:
            # Train on the given batch of Programs
            actions, obs, priors, programs = override
            for p in programs:
                Program.cache[p.str] = p


        ############################################################### 应该在这里应用 PSRN， #################### KK TODO

        # print(actions, obs, priors)
        # print('actions', 'obs', 'priors')
        print("Program.task.X_train", Program.task.X_train.shape)
        print("Program.task.y_train", Program.task.y_train.shape)

        df_pf = self.logger.get_df_pf()
        print(df_pf[["expression","complexity","r","nmse_test"]])
        pf_expressions = df_pf["expression"].tolist()
        print("pf_expressions")
        print(pf_expressions)
        
        variables_name = ["x{}".format(i+1) for i in range(Program.task.X_train.shape[1])]
        use_float_const = False
        n_psrn_tokens = 5
        n_sample_variables = min(2, len(variables_name))
        # 转 sympy -> PTS 帕累托前沿分析 -> PTS 前向得到括号表达式 -> from from_bracket_string -> Program 完成！
        n_tokens = n_psrn_tokens - n_sample_variables
        
        def break_down_expr(symbols_sympy, n_tokens):
            now_chosen_token = None
            tokens = []
            use_set = True
            MAX_LEN_SET = 1000
            SAMPLE_PROB = 0.5
            SAMPLE_PROB_CROSS_VAR = 0.5
            MAX_INTEGER = 10
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
            
            def sample_const(use_float_const):
                CONST_LB = -5
                CONST_UB = 5
                EXTRA_CONST = ["pi"]
                use_extra_const = False
                if use_float_const:
                    return round(random.uniform(CONST_LB, CONST_UB), 1)
                else:
                    CONST_LIST = np.linspace(
                        CONST_LB, CONST_UB, CONST_UB - CONST_LB + 1
                    ).tolist()
                    CONST_LIST = [x for x in CONST_LIST if x != 0]
                    if use_extra_const:
                        CONST_LIST.extend(EXTRA_CONST)
                    return random.choice(CONST_LIST)
                
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
                                    generate_cross_variable(variables_name, 1)[0]
                                )
                            else:
                                now_chosen_token = sp.S(sample_const(use_float_const))
                        else:
                            now_chosen_token = sp.S(
                                random.choices(
                                    [token for token, freq in tokens_freq],
                                    weights=[freq for token, freq in tokens_freq],
                                    k=1,
                                )[0]
                            )

                        
                        if now_chosen_token is None:
                            continue
                        if (
                            not (not use_float_const and "." in str(now_chosen_token))
                            and str(now_chosen_token) not in variables_name
                            and not has_large_integer(now_chosen_token)
                        ):
                            if now_chosen_token not in sampled_tokens:
                                sampled_tokens.append(now_chosen_token)
                            else:
                                # print('#!', end='')
                                pass
                        else:
                            # print('#!', end='')
                            pass
                else:
                    sampled_constants_num = n_tokens - len(tokens)
                    sampled_constants = [
                        sample_const(use_float_const)
                        for i in range(sampled_constants_num)
                    ]
                    # print('sampled_constants',sampled_constants)
                    sampled_tokens = tokens + sampled_constants

                # print('sampled_tokens',sampled_tokens)
                sampled_tokens = [str(t) for t in sampled_tokens]

                if use_set:
                    sampled_set = set(sampled_tokens)
                    if len(sampled_set) != len(set(sampled_tokens + variables_name)) - len(
                        variables_name
                    ):
                        continue

                    if str(sampled_set) not in visited_set:
                        visited_set.add(str(sampled_set))
                        return sampled_tokens
                    else:
                        continue
                else:
                    return sampled_tokens

            return sampled_tokens

        break_down_tokens = break_down_expr([sp.S(expr) for expr in pf_expressions], n_tokens)
        print("break_down_tokens", break_down_tokens) # ['sin(x1)', '2', 'x1**2']
        # psrn_model 
        
        sampled_variables = random.sample(variables_name, n_sample_variables)

        psrn_input_tokens = sampled_variables + break_down_tokens
        random.shuffle(psrn_input_tokens)

        expr_ls = psrn_input_tokens
        
        
        
        orginal_X = Program.task.X_train
        orginal_Y = Program.task.y_train
        
        
        
        # print("expr_ls", expr_ls)

        sampled_idx = np.unique(
            np.random.choice(
                orginal_X.shape[0],
                size=min(n_down_sample, orginal_X.shape[0]),
                replace=False,
            )
        )
        
        def get_gs_X(g_list, variables, X):

            # from utils.data import expr_to_Y_pred
            

            """get the base expressions' data (gs)

            Args:
                g_list (list): _description_
                variables (list): _description_
                X (np.ndarray): _description_

            Returns:
                Tuple[bool, np.ndarray]: success flag and gs_X array (n, n_gs), where n_gs == len(g_list)
            """
            success = False
            
            gs_X = []
            for g in g_list:
                try:
                    g_sympy = se_sympify(g)
                except RuntimeError as e:
                    g_sympy = sp.S("1")

                g_X = expr_to_Y_pred(g_sympy, X, variables_name)  # -> [n, 1]
                gs_X.append(g_X)

            # print("g_list", g_list)
            # print("gs_X", gs_X)
            gs_X = np.hstack(gs_X)
            # keep safe, np.nan or np.inf -> 0
            gs_X[np.isnan(gs_X)] = 0
            gs_X[np.isinf(gs_X)] = 0
            return success, gs_X
    
        Y = orginal_Y[sampled_idx]
        # print("expr_ls", expr_ls, "variables", variables_name)
        flag, X = get_gs_X(expr_ls, variables_name, orginal_X[sampled_idx])
        # print("flag", flag)
        X = X.real
        

        X = torch.from_numpy(X).to(device)
        Y = torch.from_numpy(Y).to(device)

        psrn_model.current_expr_ls = expr_ls
        
        
        expr_ls, _ = psrn_model.get_best_expr_and_MSE_topk(X, Y, topk)
        # print("expr_ls", expr_ls)
        # X
        # y
        
        extra_psrn_programs = []
        for e_str in expr_ls:
            try:
                # print("converting:", e_str)
                e_psrn_program = Program.from_bracket_string(e_str)
                # print("get:", e_psrn_program)
                extra_psrn_programs.append(e_psrn_program)
            except Exception as e:
                print("expr {} convert failed! in from_bracket_string".format(e_str))
            
        
        
        
        n_psrn_extra = len(extra_psrn_programs)
        if n_psrn_extra > 0:
            programs = programs[:-n_psrn_extra] + extra_psrn_programs
            
        
        ###############################################################################################################


        # Extra samples, previously already contained in cache,
        # that were geneated during the attempt to get
        # batch_size new samples for expensive reward evaluation
        if self.policy.valid_extended_batch:
            self.policy.valid_extended_batch = False
            n_extra = self.policy.extended_batch[0]
            if n_extra > 0:
                extra_programs = [from_tokens(a) for a in
                                  self.policy.extended_batch[1]]
                # Concatenation is fine because rnn_policy.sample_novel()
                # already made sure that offline batch and extended batch
                # are padded to the same trajectory length
                actions = np.concatenate([actions, self.policy.extended_batch[1]])
                obs = np.concatenate([obs, self.policy.extended_batch[2]])
                priors = np.concatenate([priors, self.policy.extended_batch[3]])
                programs = programs + extra_programs

        self.nevals += self.batch_size + n_extra + n_psrn_extra

        # Run GP seeded with the current batch, returning elite samples
        if self.gp_controller is not None:
            deap_programs, deap_actions, deap_obs, deap_priors = self.gp_controller(actions)
            self.nevals += self.gp_controller.nevals

            # Pad AOP if different sized
            if actions.shape[1] < deap_actions.shape[1]:
                # If RL shape is smaller than GP then pad
                pad_length = deap_actions.shape[1] - actions.shape[1]
                actions, obs, priors = pad_action_obs_priors(actions, obs, priors, pad_length)
            elif actions.shape[1] > deap_actions.shape[1]:
                # If GP shape is smaller than RL then pad
                pad_length = actions.shape[1] - deap_actions.shape[1]
                deap_actions, deap_obs, deap_priors = pad_action_obs_priors(deap_actions, deap_obs, deap_priors, pad_length)

            # Combine RNN and deap programs, actions, obs, and priors
            programs = programs + deap_programs
            actions = np.append(actions, deap_actions, axis=0)
            obs = np.append(obs, deap_obs, axis=0)
            priors = np.append(priors, deap_priors, axis=0)

        # Compute rewards in parallel
        if self.pool is not None:
            # Filter programs that need reward computing
            programs_to_optimize = list(set([p for p in programs if "r" not in p.__dict__]))
            pool_p_dict = { p.str : p for p in self.pool.map(work, programs_to_optimize) }
            programs = [pool_p_dict[p.str] if "r" not in p.__dict__  else p for p in programs]
            # Make sure to update cache with new programs
            Program.cache.update(pool_p_dict)

        # Compute rewards (or retrieve cached rewards)
        r = np.array([p.r for p in programs])

        # Back up programs to save them properly later
        controller_programs = programs.copy() if self.logger.save_token_count else None

        # Need for Vanilla Policy Gradient (epsilon = null)
        l           = np.array([len(p.traversal) for p in programs])
        s           = [p.str for p in programs] # Str representations of Programs
        on_policy   = np.array([p.originally_on_policy for p in programs])
        invalid     = np.array([p.invalid for p in programs], dtype=bool)

        if self.logger.save_positional_entropy:
            positional_entropy = np.apply_along_axis(empirical_entropy, 0, actions)

        if self.logger.save_top_samples_per_batch > 0:
            # sort in descending order: larger rewards -> better solutions
            sorted_idx = np.argsort(r)[::-1]
            top_perc = int(len(programs) * float(self.logger.save_top_samples_per_batch))
            for idx in sorted_idx[:top_perc]:
                top_samples_per_batch.append([self.iteration, r[idx], repr(programs[idx])])

        # Store in variables the values for the whole batch (those variables will be modified below)
        r_full = r
        l_full = l
        s_full = s
        actions_full = actions
        invalid_full = invalid
        r_max = np.max(r)





        """
        Apply risk-seeking policy gradient: compute the empirical quantile of
        rewards and filter out programs with lesser reward.
        """
        if self.epsilon is not None and self.epsilon < 1.0:
            # Compute reward quantile estimate
            if self.use_memory: # Memory-augmented quantile
                # Get subset of Programs not in buffer
                unique_programs = [p for p in programs \
                                   if p.str not in self.memory_queue.unique_items]
                N = len(unique_programs)

                # Get rewards
                memory_r = self.memory_queue.get_rewards()
                sample_r = [p.r for p in unique_programs]
                combined_r = np.concatenate([memory_r, sample_r])

                # Compute quantile weights
                memory_w = self.memory_queue.compute_probs()
                if N == 0:
                    print("WARNING: Found no unique samples in batch!")
                    combined_w = memory_w / memory_w.sum() # Renormalize
                else:
                    sample_w = np.repeat((1 - memory_w.sum()) / N, N)
                    combined_w = np.concatenate([memory_w, sample_w])

                # Quantile variance/bias estimates
                if self.memory_threshold is not None:
                    print("Memory weight:", memory_w.sum())
                    if memory_w.sum() > self.memory_threshold:
                        quantile_variance(self.memory_queue, self.policy, self.batch_size, self.epsilon, self.iteration)

                # Compute the weighted quantile
                quantile = weighted_quantile(values=combined_r, weights=combined_w, q=1 - self.epsilon)

            else: # Empirical quantile
                quantile = np.quantile(r, 1 - self.epsilon, interpolation="higher")

            # Filter quantities whose reward >= quantile
            keep = r >= quantile
            l = l[keep]
            s = list(compress(s, keep))
            invalid = invalid[keep]
            r = r[keep]
            programs  = list(compress(programs, keep))
            actions = actions[keep, :]
            obs = obs[keep, :, :]
            priors = priors[keep, :, :]
            on_policy = on_policy[keep]

        # Clip bounds of rewards to prevent NaNs in gradient descent
        r = np.clip(r, -1e6, 1e6)

        # Compute baseline
        # NOTE: pg_loss = tf.reduce_mean((self.r - self.baseline) * neglogp, name="pg_loss")
        if self.baseline == "ewma_R":
            ewma = np.mean(r) if ewma is None else self.alpha*np.mean(r) + (1 - self.alpha)*ewma
            b = ewma
        elif self.baseline == "R_e": # Default
            ewma = -1
            b = quantile
        elif self.baseline == "ewma_R_e":
            ewma = np.min(r) if ewma is None else self.alpha*quantile + (1 - self.alpha)*ewma
            b = ewma
        elif self.baseline == "combined":
            ewma = np.mean(r) - quantile if ewma is None else self.alpha*(np.mean(r) - quantile) + (1 - self.alpha)*ewma
            b = quantile + ewma

        # Compute sequence lengths
        lengths = np.array([min(len(p.traversal), self.policy.max_length)
                            for p in programs], dtype=np.int32)

        # Create the Batch
        sampled_batch = Batch(actions=actions, obs=obs, priors=priors,
                              lengths=lengths, rewards=r, on_policy=on_policy)

        # Update and sample from the priority queue 
        if self.priority_queue is not None:
            self.priority_queue.push_best(sampled_batch, programs)
            pqt_batch = self.priority_queue.sample_batch(self.policy_optimizer.pqt_batch_size)
            # Train the policy
            summaries = self.policy_optimizer.train_step(b, sampled_batch, pqt_batch)
        else:
            pqt_batch = None
            # Train the policy
            summaries = self.policy_optimizer.train_step(b, sampled_batch)

        # Walltime calculation for the iteration
        iteration_walltime = time.time() - start_time

        # Update the memory queue
        if self.memory_queue is not None:
            self.memory_queue.push_batch(sampled_batch, programs)

        # Update new best expression
        if r_max > self.r_best:
            self.r_best = r_max
            self.p_r_best = programs[np.argmax(r)]             #  # KK TODO program 就在这里进行更新的

            # Print new best expression
            if self.verbose or self.debug:
                print("[{}] Training iteration {}, current best R: {:.4f}".format(get_duration(start_time), self.iteration + 1, self.r_best))
                print("\n\t** New best (in train.py)")
                self.p_r_best.print_stats()

        # Collect sub-batch statistics and write output
        self.logger.save_stats(r_full, l_full, actions_full, s_full,
                               invalid_full, r, l, actions, s, s_history,
                               invalid, self.r_best, r_max, ewma, summaries,
                               self.iteration, b, iteration_walltime,
                               self.nevals, controller_programs,
                               positional_entropy, top_samples_per_batch)


        # Stop if early stopping criteria is met
        if self.early_stopping and self.p_r_best.evaluate.get("success"):
            print("[{}] Early stopping criteria met; breaking early.".format(get_duration(start_time)))
            self.done = True

        if self.verbose and (self.iteration + 1) % 10 == 0:
            print("[{}] Training iteration {}, current best R: {:.4f}".format(get_duration(start_time), self.iteration + 1, self.r_best))

        if self.debug >= 2:
            print("\nParameter means after iteration {}:".format(self.iteration + 1))
            self.print_var_means()

        if self.nevals >= self.n_samples:
            self.done = True

        # Increment the iteration counter
        self.iteration += 1


        # print('='*40)
        # print("programs:")        # KK TODO program 就在这里，
        # print(programs)
        # print('='*40)
        

    def save(self, save_path):
        """
        Save the state of the Trainer.
        """

        state_dict = {
            "nevals" : self.nevals,
            "iteration" : self.iteration,
            "r_best" : self.r_best,
            "p_r_best_tokens" : self.p_r_best.tokens.tolist() if self.p_r_best is not None else None
        }
        with open(save_path, 'w') as f:
            json.dump(state_dict, f)

        print("Saved Trainer state to {}.".format(save_path))

    def load(self, load_path):
        """
        Load the state of the Trainer.
        """

        with open(load_path, 'r') as f:
            state_dict = json.load(f)

        # Load nevals and iteration from savestate
        self.nevals = state_dict["nevals"]
        self.iteration = state_dict["iteration"]

        # Load r_best and p_r_best
        if state_dict["p_r_best_tokens"] is not None:
            tokens = np.array(state_dict["p_r_best_tokens"], dtype=np.int32)
            self.p_r_best = from_tokens(tokens)
        else:
            self.p_r_best = None

        print("Loaded Trainer state from {}.".format(load_path))
