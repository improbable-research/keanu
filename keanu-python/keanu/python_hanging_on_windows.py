import sys
import io
import os.path
from py4j.java_gateway import JavaGateway, JavaObject, CallbackServerParameters

PATH = os.path.abspath(os.path.dirname(__file__))
ND4J_CLASSPATH_ENVIRONMENT_VARIABLE = "KEANU_ND4J_CLASSPATH"

CLASSPATH = os.path.join(PATH, "classpath", "*")
nd4j_path = os.environ.get(ND4J_CLASSPATH_ENVIRONMENT_VARIABLE)
if nd4j_path is not None:
    CLASSPATH = os.pathsep.join([CLASSPATH, os.path.join(nd4j_path, "*")])

print("classpath=", CLASSPATH)

gateway = JavaGateway.launch_gateway(
    classpath=CLASSPATH,
    die_on_exit=True
)
print("DoubleTensor.scalar(1.0)")
t = gateway.new_jvm_view().io.improbable.keanu.tensor.dbl.DoubleTensor.scalar(1.0)
print(t)

print("DoubleTensor.eye(1)")
t = gateway.new_jvm_view().io.improbable.keanu.tensor.dbl.DoubleTensor.eye(1)
print(t)

print("DoubleTensor.create(1.0)")
arr = gateway.new_array(gateway.jvm.double, 1)
arr[0] = 1.0
t = gateway.new_jvm_view().io.improbable.keanu.tensor.dbl.DoubleTensor.create(arr)
print(t)

print("DoubleTensor.create(1.0, 2.0, 3.0)")
arr = gateway.new_array(gateway.jvm.double, 3)
arr[0] = 1.0
arr[1] = 2.0
arr[2] = 3.0
t = gateway.new_jvm_view().io.improbable.keanu.tensor.dbl.DoubleTensor.create(arr)
print(t)    # this one is hanging
