import torch
import torch.distributions as dist

from daphne import daphne

from primitives import funcprimitives #TODO
from tests import is_tol, run_prob_test,load_truth

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
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


def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        return env[op](*map(deterministic_eval, args))
    elif type(exp) is int or type(exp) is float:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
    else:
        raise("Expression type unknown.", exp)

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
        
        elif prim == "sample*":
            return evaluate(exp[1], loc_env, procedure_defs).sample()
        
        elif prim == "if":
            e_prime = evaluate(exp[1], loc_env, procedure_defs)
            if e_prime:
                return evaluate(exp[2], loc_env, procedure_defs)
            return evaluate(exp[3], loc_env, procedure_defs)

        # we do not implement for now
        elif prim == "observe*":
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
                for i, var in enumerate(procedure_defs[prim][1]):
                    loc_env[var] = eval_exps[i]
                e0_prime = procedure_defs[prim][2]
                return evaluate(e0_prime, loc_env, procedure_defs)

            else:
                return eval_exps 

    else:
        raise("Expression type unknown.", exp)

             
def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    
    fn_defs = graph[0]
    body = graph[1]
    E = graph[2]

    all_vars = body["V"]
    dependencies = body["A"]
    link_fns = body["P"]
    side_effects = body["Y"]
    
    # TODO(innaivanova): do not actually depend on the naming of the parent vars
    sorted_parents = sorted(dependencies.keys(), key=lambda x: int(x[6:]))
    local_env = {}

    for parent in sorted_parents:
        local_env[parent] = evaluate(link_fns[parent], local_env, fn_defs)
        
    for var, exp in link_fns.items():
        local_env[var] = evaluate(exp, local_env, fn_defs)

    return evaluate(E, local_env, fn_defs)



def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)   



#Testing:

def run_deterministic_tests():
    
    for i in range(1,13):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    #TODO: 
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        graph = daphne(['graph', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


        
        
if __name__ == '__main__':
    

    run_deterministic_tests()
    run_probabilistic_tests()


    for i in range(1,5):
        graph = daphne(['graph','-i','../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(sample_from_joint(graph))    

    
