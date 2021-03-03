import torch

def vector(*arg):
    # general case
    try:
        return torch.stack(arg, dim=0)
    
    # for concatenation of many vectors
    except RuntimeError:
        dim = len(arg[0].shape) - 1
        return torch.cat(arg, dim=dim)
    
    # for distribution objects
    except TypeError:
        return list(arg)

def get(v, i):
    return v[int(i.item())]

def put(v, i, c):
    v[int(i.item())] = c
    return v

def first(v):
    return v[0]

def second(v):
    return v[1]

def last(v):
    return v[-1]

def append(v, c):
    return torch.cat((v, c.unsqueeze(dim=0)), dim=0)

def hashmap(*v):
    hm = {}
    i = 0
    while i < len(v):
        hm[v[i].item()] = v[i+1]
        i+=2
    return hm

def less_than(*args):
    return args[0] < args[1]

def rest(v):
    return v[1:]

def l(*arg):
    return list(arg)

def cons(x, l):
    return [x] + l    

funcprimitives = {
    "vector": vector,
    "get": get,
    "put": put,
    "first": first,
    "last": last,
    "append": append,
    "hash-map": hashmap,
    "less_than": less_than,
    "second": second,
    "rest": rest,
    "conj": append,
    "list": l,
    "cons": cons,
}
