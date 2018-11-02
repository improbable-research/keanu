class Function:
    def __init__(self, lambda_function):
        self.lambda_function = lambda_function

    def apply(self, arg):
        return self.lambda_function(arg)

    class Java:
        implements = ["java.util.function.Function"]
