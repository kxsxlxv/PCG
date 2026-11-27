"""Bounded geometry additions, never a mandate to exhaust the cap."""
HARD_CAP=1800
ENTRY_ALLOWANCE=96
EQUIPMENT_ALLOWANCE=144

def ceiling(base, allowance):
    if type(base) is not int or not 0<=base<=HARD_CAP:
        raise ValueError('invalid base triangle count')
    if type(allowance) is not int or allowance<0:
        raise ValueError('invalid allowance')
    return min(HARD_CAP,base+allowance)

def entry_budget(base, entries):
    if type(entries) is not int or entries<0:
        raise ValueError('invalid entry count')
    return ceiling(base,min(ENTRY_ALLOWANCE,16*entries))-base

def equipment_ceiling(base):
    return ceiling(base,EQUIPMENT_ALLOWANCE)
