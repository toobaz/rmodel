from .exceptions import MissingModelResult

class FakeNumber:
    """
    Produce an object that will raise an error when transformed to any integer,
    float or string representation.
    """
    def __init__(self, msg):
        self._msg = msg

for attr in ('round', 'sqrt', '__mul__', '__round__', '__add__'):
    # Arithmetic operations just propagate, like if this was a NaN
    setattr(FakeNumber, attr, lambda self, *args, **kwargs : self)
for attr in ('__int__', '__float__', '__str__'):
    # Raise when attempting to actually display:
    def protest(self, *args, **kwargs):
        raise MissingModelResult(self._msg)
    setattr(FakeNumber, attr, protest)

