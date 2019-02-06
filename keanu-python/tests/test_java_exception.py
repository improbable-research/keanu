import pytest
from py4j.protocol import Py4JJavaError

from keanu.context import KeanuContext
from keanu.java_exception import JavaException


def test_you_can_get_info_from_a_java_exception() -> None:
    context = KeanuContext()
    with pytest.raises(Py4JJavaError) as excinfo:
        context.jvm_view().java.util.HashMap(-1)

    java_exception = JavaException(excinfo.value)
    assert java_exception.get_name() == "java.lang.IllegalArgumentException"
    assert java_exception.get_message() == "Illegal initial capacity: -1"
    assert java_exception.unwrap().getCause() == None


def test_you_can_throw_a_java_exception() -> None:
    with pytest.raises(JavaException, match="Illegal initial capacity: -1"):
        context = KeanuContext()
        try:
            context.jvm_view().java.util.HashMap(-1)
        except Py4JJavaError as e:
            raise JavaException(e)


def test_its_repr_method_gives_you_the_stack_trace():
    expected_string = """An error occurred while calling None.java.util.HashMap.
: java.lang.IllegalArgumentException: Illegal initial capacity: -1
	at java.util.HashMap.<init>(HashMap.java:449)
	at java.util.HashMap.<init>(HashMap.java:468)
	at sun.reflect.NativeConstructorAccessorImpl.newInstance0(Native Method)
	at sun.reflect.NativeConstructorAccessorImpl.newInstance(NativeConstructorAccessorImpl.java:62)
	at sun.reflect.DelegatingConstructorAccessorImpl.newInstance(DelegatingConstructorAccessorImpl.java:45)
	at java.lang.reflect.Constructor.newInstance(Constructor.java:423)
	at py4j.reflection.MethodInvoker.invoke(MethodInvoker.java:247)
	at py4j.reflection.ReflectionEngine.invoke(ReflectionEngine.java:357)
	at py4j.Gateway.invoke(Gateway.java:238)
	at py4j.commands.ConstructorCommand.invokeConstructor(ConstructorCommand.java:80)
	at py4j.commands.ConstructorCommand.execute(ConstructorCommand.java:69)
	at py4j.GatewayConnection.run(GatewayConnection.java:238)
	at java.lang.Thread.run(Thread.java:748)
"""

    context = KeanuContext()
    with pytest.raises(Py4JJavaError) as excinfo:
        context.jvm_view().java.util.HashMap(-1)

    java_exception = JavaException(excinfo.value)

    assert str(java_exception) == expected_string


