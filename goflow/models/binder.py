from . import *
from hailhydro import *

modelBinder ={
    'default': murray.murray,
    'murray': murray.murray,
    'bohn': bohn.bohn,
    'corson': corson.corson,
    'meigel': meigel.meigel,
    'link': meigel.link,
    'kramer': kramer.kramer,
    # 'volume': meigel.volume,
}
circuitBinder = {
    'murray': flow_init.Flow,
    'bohn': flow_init.Flow,
    'corson': flow_random.FlowRandom,
    'meigel': flux_overflow.Overflow,
    'link': flux_overflow.Overflow,
    'kramer': kramer.dualFlowRandom,
    # 'volume': flux_overflow.Overflow,
}
