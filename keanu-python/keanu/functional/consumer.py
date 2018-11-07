class Consumer:
    def __init__(self, lambda_function):
        self.lambda_function = lambda_function

    def accept(self, arg):
        self.lambda_function(arg)

    class Java:
        implements = ["java.util.function.Consumer"]