"""Validate dedicated decision replies before accepting fresh or replayed output."""
import math
CALIBRATION_VERSION='p3-r2-s000291-authored'

def validate_decision(result,options):
    def fail():raise ValueError('invalid dedicated decision result')
    if not isinstance(result,dict) or not isinstance(options,list) or not 2<=len(options)<=7 or any(type(x)is not str or not x or x=='__unknown__' for x in options) or len(set(options))!=len(options):fail()
    if not {'value','abstained','probabilities','unknown_probability','raw_logits','calibration_version','instructions'}<=result.keys():fail()
    if type(result.get('abstained'))is not bool:fail()
    value=result.get('value')
    if value is not None and (type(value)is not str or value not in options):fail()
    if result['abstained']!=(value is None):fail()
    def numbers(values,count,probability=False):
        if type(values)is not list or len(values)!=count:fail()
        for v in values:
            if type(v)not in (int,float) or probability and not 0<=v<=1:fail()
            try:finite=math.isfinite(v)
            except OverflowError:fail()
            if not finite:fail()
    numbers(result.get('probabilities'),len(options),True)
    numbers(result.get('raw_logits'),len(options)+1)
    numbers([result.get('unknown_probability')],1,True)
    if abs(math.fsum(result['probabilities'])+result['unknown_probability']-1)>1e-5:fail()
    if result.get('calibration_version')!=CALIBRATION_VERSION:fail()
    if type(result.get('instructions'))is not int or result['instructions']<0:fail()
    return result
