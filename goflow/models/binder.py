# @Author: Felix Kramer <felix>
# @Date:   2022-06-28T16:24:41+02:00
# @Email:  felixuwekramer@proton.me
# @Filename: binder.py
# @Last modified by:   felix
# @Last modified time: 2022-07-01T15:12:56+02:00

from . import murray, bohn, corson, meigel, kramer
from hailhydro import flow_init, flow_random, flux_overflow

modelBinder = {
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
