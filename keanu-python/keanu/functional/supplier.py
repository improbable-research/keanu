class Supplier:
    def __init__(self, lambda_function):
        self.lambda_function = lambda_function

    def get(self):
        return self.lambda_function()

    class Java:
        implements = ["java.util.function.Supplier"]