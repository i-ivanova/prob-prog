import torch
from torch import distributions as dist
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from primitives import funcprimitives

env = {'normal': dist.Normal,
       'sqrt': torch.sqrt,
       '+': torch.add,
       '-': torch.sub,
       '*': torch.mul,
       '/': torch.div,
       'beta': dist.Beta,
       'gamma': dist.Gamma,
       'dirichlet': dist.Dirichlet,
       'exponential': dist.Exponential,
       'discrete': dist.Categorical,
       'uniform': dist.Uniform,
       'uniform-continuous': dist.Uniform,
       'vector': funcprimitives["vector"],
       'get': funcprimitives["get"],
       'put': funcprimitives["put"],
       'hash-map': funcprimitives["hash-map"],
       'first': funcprimitives["first"],
       'second': funcprimitives["second"],
       'last': funcprimitives["last"],
       'append': funcprimitives["append"],
       'conj': funcprimitives["append"],
       '<': funcprimitives["less_than"],
       'mat-mul': torch.matmul,
       'mat-repmat': lambda x, y, z: x.repeat((int(y.item()), int(z.item()))),
       'mat-add': torch.add,
       'mat-tanh': torch.tanh,
       'mat-transpose': torch.t,
       'rest': funcprimitives["rest"],
      }

def evaluate(exp, loc_env, procedure_defs):
#     print(exp, loc_env, "\n------------------------------------------")

    if type(exp) is int or type(exp) is float:
        return torch.tensor(float(exp))
    
    elif type(exp) is str:
        return loc_env[exp]

    elif type(exp) is list:
        prim = exp[0]
        
        if prim == "let":
            loc_env[exp[1][0]] = evaluate(exp[1][1], loc_env, procedure_defs)
            return evaluate(exp[2], loc_env, procedure_defs)
        
        elif prim == "sample":
            return evaluate(exp[1], loc_env, procedure_defs).sample()
        
        elif prim == "if":
            e_prime = evaluate(exp[1], loc_env, procedure_defs)
            if e_prime:
                return evaluate(exp[2], loc_env, procedure_defs)
            return evaluate(exp[3], loc_env, procedure_defs)

        # we do not implement for now
        elif prim == "observe":
            return torch.tensor(0.0)
        
        elif prim in env:
            # need this otherwise loc_env is lost
            eval_exp = [evaluate(x, loc_env, procedure_defs) for x in exp[1:]]
            return env[prim](*eval_exp)

        else:
            # evaluate the rest of the input expresions
            eval_exps = [evaluate(e, loc_env, procedure_defs) for e in exp[1:]]
           
            if prim in procedure_defs:    
                # iterate over function variables
                for i, var in enumerate(procedure_defs[exp[0]]["vars"]):
                    loc_env[var] = eval_exps[i]
                e0_prime = procedure_defs[prim]["fn_exp"]
                return evaluate(e0_prime, loc_env, procedure_defs)

            else:
                return eval_exps 

    else:
        raise("Expression type unknown.", exp)

    

def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    copy_ast = ast[:]
    loc_env = {}
    procedure_defs = {}
    
    for exp in copy_ast:
        if exp[0] == "defn":
            defn = {"vars": exp[2], "fn_exp": exp[3]}
            procedure_defs[exp[1]] = defn

    return evaluate(copy_ast[-1], loc_env, procedure_defs), {}  # empty sigma

def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)[0]
    


def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    

        
if __name__ == '__main__':

    run_deterministic_tests()
    
    run_probabilistic_tests()


    for i in range(1,5):
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast)[0])
