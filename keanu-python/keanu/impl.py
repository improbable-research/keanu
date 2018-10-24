class UnaryLambda:
    def __init__(self, fn):
        self._fn = fn

    def apply(self, val):
        return self._fn(val)

    class Java:
        implements = ["java.util.function.Function"]
