package io.improbable.keanu.backend.keanu.compiled;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.VertexUnaryOp;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.BooleanIfVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.BooleanProxyVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.CastToBooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NumericalEqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.AndBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.OrBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.XorBinaryVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.EqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanOrEqualVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.BooleanConcatenationVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BooleanReshapeVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BooleanSliceVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BooleanTakeVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.NotBinaryVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.CastNumberToDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleIfVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.ArcTan2Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DifferenceVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DivisionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleGreaterThanMaskVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleGreaterThanOrEqualToMaskVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleLessThanMaskVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleLessThanOrEqualToMaskVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MatrixMultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MaxVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MinVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.ternary.DoubleSetWithMaskVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.AbsVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ArcCosVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ArcSinVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ArcTanVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.CeilVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.CosVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ExpVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.FloorVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.LogGammaVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.LogVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.MatrixDeterminantVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.MatrixInverseVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.PermuteVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ReshapeVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.RoundVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SigmoidVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SinVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SliceVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SumVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.TakeVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.TanVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.IfVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.MultiplexerVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.PrintVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.GenericSliceVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.GenericTakeVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.CastNumberToIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.IntegerIfVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.IntegerProxyVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerAdditionVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDifferenceVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDivisionVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerGetBooleanIndexVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMaxVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMinVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMultiplicationVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerPowerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.multiple.IntegerConcatenationVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerAbsVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerBroadcastVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerPermuteVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerReshapeVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerSliceVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerSumVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerTakeVertex;
import io.improbable.keanu.vertices.utility.AssertVertex;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * This class provides a static lookup per vertex to a string that represents the right hand side the assignment
 * in compiled graph. This should include everything up to but not including the semicolon that ends the line of code
 * that is the complete assignment.
 * <p>
 * e.g. AdditionVertex.class -&gt;
 * <p>
 * leftArg &lt;= left hand arg from additionVertex
 * rightArg &lt;= right hand arg from additionVertex
 * <p>
 * lookup leftArg and rightArg
 * return leftArg + ".plus(" + rightArg + ")"
 */
public class KeanuVertexToTensorOpMapper {

    public static final boolean ENABLE_IN_PLACE = true;

    private static Map<Class<?>, OpMapper> opMappers;

    static {
        opMappers = new HashMap<>();

        //Double ops
        opMappers.put(AdditionVertex.class, fluentBinaryOp("plus", "plusInPlace"));
        opMappers.put(DifferenceVertex.class, fluentBinaryOp("minus", "minusInPlace"));
        opMappers.put(DivisionVertex.class, fluentBinaryOp("div", "divInPlace"));
        opMappers.put(MultiplicationVertex.class, fluentBinaryOp("times", "timesInPlace"));
        opMappers.put(MatrixMultiplicationVertex.class, fluentBinaryOp("matrixMultiply"));
        opMappers.put(PowerVertex.class, fluentBinaryOp("pow", "powInPlace"));
        opMappers.put(ArcTan2Vertex.class, fluentBinaryOp("atan2", "atan2InPlace"));

        opMappers.put(DoubleGreaterThanOrEqualToMaskVertex.class, fluentBinaryOp("greaterThanOrEqualToMask"));
        opMappers.put(DoubleGreaterThanMaskVertex.class, fluentBinaryOp("greaterThanMask"));
        opMappers.put(DoubleLessThanOrEqualToMaskVertex.class, fluentBinaryOp("lessThanOrEqualToMask"));
        opMappers.put(DoubleLessThanMaskVertex.class, fluentBinaryOp("lessThanMask"));

        opMappers.put(DoubleSetWithMaskVertex.class, KeanuVertexToTensorOpMapper::setWithMaskDoubleOp);

        opMappers.put(AbsVertex.class, fluentUnaryOp("abs", "absInPlace"));
        opMappers.put(CosVertex.class, fluentUnaryOp("cos", "cosInPlace"));
        opMappers.put(ArcCosVertex.class, fluentUnaryOp("acos", "acosInPlace"));
        opMappers.put(ExpVertex.class, fluentUnaryOp("exp", "expInPlace"));
        opMappers.put(LogVertex.class, fluentUnaryOp("log", "logInPlace"));
        opMappers.put(LogGammaVertex.class, fluentUnaryOp("logGamma", "logGammaInPlace"));
        opMappers.put(SinVertex.class, fluentUnaryOp("sin", "sinInPlace"));
        opMappers.put(ArcSinVertex.class, fluentUnaryOp("asin", "asinInPlace"));
        opMappers.put(TanVertex.class, fluentUnaryOp("tan", "tanInPlace"));
        opMappers.put(ArcTanVertex.class, fluentUnaryOp("atan", "atanInPlace"));
        opMappers.put(CeilVertex.class, fluentUnaryOp("ceil", "ceilInPlace"));
        opMappers.put(FloorVertex.class, fluentUnaryOp("floor", "floorInPlace"));
        opMappers.put(RoundVertex.class, fluentUnaryOp("round", "roundInPlace"));
        opMappers.put(SigmoidVertex.class, fluentUnaryOp("sigmoid", "sigmoidInPlace"));

        opMappers.put(MatrixDeterminantVertex.class, fluentUnaryOp("determinant"));
        opMappers.put(MatrixInverseVertex.class, fluentUnaryOp("matrixInverse"));

        opMappers.put(ConcatenationVertex.class, KeanuVertexToTensorOpMapper::concatDoubleOp);
        opMappers.put(SumVertex.class, KeanuVertexToTensorOpMapper::sumDoubleOp);
        opMappers.put(ReshapeVertex.class, KeanuVertexToTensorOpMapper::reshapeDoubleOp);
        opMappers.put(PermuteVertex.class, KeanuVertexToTensorOpMapper::permuteDoubleOp);
        opMappers.put(SliceVertex.class, KeanuVertexToTensorOpMapper::sliceDoubleOp);
        opMappers.put(TakeVertex.class, KeanuVertexToTensorOpMapper::takeDoubleOp);

        opMappers.put(MaxVertex.class, fluentBinaryOp("max"));
        opMappers.put(MinVertex.class, fluentBinaryOp("min"));

        opMappers.put(CastNumberToDoubleVertex.class, fluentUnaryOp("toDouble"));

        opMappers.put(DoubleIfVertex.class, KeanuVertexToTensorOpMapper::doubleIfOp);
        opMappers.put(DoubleProxyVertex.class, KeanuVertexToTensorOpMapper::doubleProxyOp);

        //Integer ops
        opMappers.put(IntegerAbsVertex.class, fluentUnaryOp("abs"));

        opMappers.put(IntegerMultiplicationVertex.class, fluentBinaryOp("times", "timesInPlace"));
        opMappers.put(IntegerAdditionVertex.class, fluentBinaryOp("plus", "plusInPlace"));
        opMappers.put(IntegerDifferenceVertex.class, fluentBinaryOp("minus", "minusInPlace"));
        opMappers.put(IntegerDivisionVertex.class, fluentBinaryOp("divideBy", "divideByInPlace"));
        opMappers.put(IntegerPowerVertex.class, fluentBinaryOp("pow", "powInPlace"));

        opMappers.put(IntegerConcatenationVertex.class, KeanuVertexToTensorOpMapper::concatIntegerOp);
        opMappers.put(IntegerSumVertex.class, KeanuVertexToTensorOpMapper::sumIntegerOp);
        opMappers.put(IntegerReshapeVertex.class, KeanuVertexToTensorOpMapper::reshapeIntegerOp);
        opMappers.put(IntegerSliceVertex.class, KeanuVertexToTensorOpMapper::sliceIntegerOp);
        opMappers.put(IntegerPermuteVertex.class, KeanuVertexToTensorOpMapper::permuteIntegerOp);
        opMappers.put(IntegerTakeVertex.class, KeanuVertexToTensorOpMapper::takeIntegerOp);
        opMappers.put(IntegerGetBooleanIndexVertex.class, fluentBinaryOp("get"));
        opMappers.put(IntegerBroadcastVertex.class, KeanuVertexToTensorOpMapper::broadcastIntegerOp);

        opMappers.put(IntegerMaxVertex.class, fluentBinaryOp("max"));
        opMappers.put(IntegerMinVertex.class, fluentBinaryOp("min"));

        opMappers.put(CastNumberToIntegerVertex.class, fluentUnaryOp("toInteger"));

        opMappers.put(IntegerIfVertex.class, KeanuVertexToTensorOpMapper::integerIfOp);
        opMappers.put(IntegerVertex.IntegerDiagVertex.class, fluentUnaryOp("diag"));
        opMappers.put(IntegerProxyVertex.class, KeanuVertexToTensorOpMapper::integerProxyOp);

        //Booleans ops
        opMappers.put(BooleanConcatenationVertex.class, KeanuVertexToTensorOpMapper::concatBoolOp);
        opMappers.put(BooleanReshapeVertex.class, KeanuVertexToTensorOpMapper::reshapeBooleanOp);
        opMappers.put(BooleanSliceVertex.class, KeanuVertexToTensorOpMapper::sliceBooleanOp);
        opMappers.put(BooleanTakeVertex.class, KeanuVertexToTensorOpMapper::takeBooleanOp);

        opMappers.put(GreaterThanOrEqualVertex.class, fluentBinaryOp("greaterThanOrEqual"));
        opMappers.put(GreaterThanVertex.class, fluentBinaryOp("greaterThan"));
        opMappers.put(LessThanOrEqualVertex.class, fluentBinaryOp("lessThanOrEqual"));
        opMappers.put(LessThanVertex.class, fluentBinaryOp("lessThan"));
        opMappers.put(EqualsVertex.class, fluentBinaryOp("elementwiseEquals"));
        opMappers.put(NotEqualsVertex.class, KeanuVertexToTensorOpMapper::notOp);
        opMappers.put(NumericalEqualsVertex.class, KeanuVertexToTensorOpMapper::numericalEqualsOp);

        opMappers.put(OrBinaryVertex.class, fluentBinaryOp("or"));
        opMappers.put(AndBinaryVertex.class, fluentBinaryOp("and"));
        opMappers.put(NotBinaryVertex.class, fluentUnaryOp("not"));
        opMappers.put(XorBinaryVertex.class, fluentUnaryOp("xor"));

        opMappers.put(CastToBooleanVertex.class, fluentUnaryOp("toBoolean"));

        opMappers.put(BooleanVertex.BooleanDiagVertex.class, fluentUnaryOp("diag"));
        opMappers.put(BooleanVertex.BooleanToIntegerMaskVertex.class, fluentUnaryOp("toIntegerMask"));
        opMappers.put(BooleanVertex.BooleanToDoubleMaskVertex.class, fluentUnaryOp("toDoubleMask"));

        opMappers.put(BooleanIfVertex.class, KeanuVertexToTensorOpMapper::booleanIfOp);
        opMappers.put(BooleanProxyVertex.class, KeanuVertexToTensorOpMapper::booleanProxyOp);

        //Constants
        opMappers.put(ConstantIntegerVertex.class, KeanuVertexToTensorOpMapper::constant);
        opMappers.put(ConstantDoubleVertex.class, KeanuVertexToTensorOpMapper::constant);
        opMappers.put(ConstantBooleanVertex.class, KeanuVertexToTensorOpMapper::constant);

        //Generic ops
        opMappers.put(IfVertex.class, KeanuVertexToTensorOpMapper::genericIfOp);
        opMappers.put(GenericSliceVertex.class, KeanuVertexToTensorOpMapper::sliceGenericOp);
        opMappers.put(GenericTakeVertex.class, KeanuVertexToTensorOpMapper::takeGenericOp);
        opMappers.put(MultiplexerVertex.class, KeanuVertexToTensorOpMapper::multiplexerOp);

        //Debug ops
        opMappers.put(PrintVertex.class, KeanuVertexToTensorOpMapper::printOp);
        opMappers.put(AssertVertex.class, KeanuVertexToTensorOpMapper::assertOp);
    }

    interface OpMapper {
        /**
         * @param vertex the operation (e.g. times, plus)
         * @param lookup lookup other variable names and any metadata about them (e.g. mutable)
         * @return the right hand side of the assignment
         */
        String apply(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup);
    }

    public static OpMapper getOpMapperFor(Class<?> clazz) {
        return opMappers.get(clazz);
    }

    private static OpMapper fluentBinaryOp(String methodName) {
        return fluentBinaryOp(methodName, methodName);
    }

    /**
     * Assumes that the left hand arg has a method called methodName and a method called
     * inPlaceMethodName. The inPlaceMethodName is used if the variable is considered mutable. It is considered
     * mutable if it is not an input, constant, or output.
     *
     * @param methodName        the method to call when not mutating the left hand variable of the binary op
     * @param inPlaceMethodName the method to call when mutating the left hand variable of the binary op
     * @return a OpMapper that provides the right hand side of an assignment. e.g. v1.plus(v2)
     */
    private static OpMapper fluentBinaryOp(String methodName, String inPlaceMethodName) {
        return (vertex, lookup) -> {
            VertexBinaryOp<?, ?> binaryOpVertex = (VertexBinaryOp<?, ?>) vertex;
            Vertex<?> left = binaryOpVertex.getLeft();
            Vertex<?> right = binaryOpVertex.getRight();

            KeanuCompiledVariable leftVariable = lookup.get(left.getReference());
            KeanuCompiledVariable rightVariable = lookup.get(right.getReference());
            boolean doInPlace = leftVariable.isMutable() && isLastChildByTopographicalSort(vertex, left) && ENABLE_IN_PLACE;
            String call = doInPlace ? inPlaceMethodName : methodName;

            return leftVariable.getName() + "." + call + "(" + rightVariable.getName() + ")";

        };
    }

    private static OpMapper fluentUnaryOp(String methodName) {
        return fluentUnaryOp(methodName, methodName);
    }

    /**
     * Assumes that the left hand arg has a method called methodName and a method called
     * inPlaceMethodName. The inPlaceMethodName is used if the variable is considered mutable. It is considered
     * mutable if it is not an input, constant, or output.
     *
     * @param methodName        the method to call when not mutating the variable of the unary op
     * @param inPlaceMethodName the method to call when mutating the variable of the unary op
     * @return a OpMapper that provides the right hand side of an assignment. e.g. v1.cos()
     */
    private static OpMapper fluentUnaryOp(String methodName, String inPlaceMethodName) {
        return (vertex, lookup) -> {
            VertexUnaryOp unaryOpVertex = (VertexUnaryOp) vertex;
            Vertex<?> input = unaryOpVertex.getInputVertex();

            KeanuCompiledVariable inputVariable = lookup.get(input.getReference());
            boolean doInPlace = inputVariable.isMutable() && isLastChildByTopographicalSort(vertex, input) && ENABLE_IN_PLACE;

            String call = doInPlace ? inPlaceMethodName : methodName;

            return inputVariable.getName() + "." + call + "()";
        };
    }

    /**
     * If the child is the last child by topographic sort order of the parent then it can reuse the memory of its
     * parent.
     *
     * @param child  the child vertex in question
     * @param parent the parent vertex in question
     * @return true if the child is the last child of the parent by topographic sort order
     */
    private static boolean isLastChildByTopographicalSort(Vertex<?> child, Vertex<?> parent) {
        Optional<Vertex> last = parent.getChildren().stream()
            .max(Comparator.comparing(Vertex::getId));

        return last
            .map(v -> v.getId().equals(child.getId()))
            .orElse(false);
    }

    private static String constant(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        throw new IllegalArgumentException("Constant should not be operation mapped");
    }

    private static String setWithMaskDoubleOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        DoubleSetWithMaskVertex setWithMaskVertex = (DoubleSetWithMaskVertex) vertex;
        Vertex mask = setWithMaskVertex.getMask();
        Vertex operand = setWithMaskVertex.getOperand();
        Vertex setValue = setWithMaskVertex.getSetValue();

        String operandName = lookup.get(operand.getId()).getName();
        String maskName = lookup.get(mask.getId()).getName();
        String setValueName = lookup.get(setValue.getId()).getName();

        return operandName + ".setWithMask(" + maskName + "," + setValueName + ".scalar())";
    }

    private static String reshapeDoubleOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        ReshapeVertex reshapeVertex = (ReshapeVertex) vertex;
        return reshapeOp(reshapeVertex.getProposedShape(), reshapeVertex.getInputVertex(), lookup);
    }

    private static String reshapeIntegerOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        IntegerReshapeVertex reshapeVertex = (IntegerReshapeVertex) vertex;
        return reshapeOp(reshapeVertex.getProposedShape(), reshapeVertex.getInputVertex(), lookup);
    }

    private static String reshapeBooleanOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        BooleanReshapeVertex reshapeVertex = (BooleanReshapeVertex) vertex;
        return reshapeOp(reshapeVertex.getProposedShape(), reshapeVertex.getInputVertex(), lookup);
    }

    private static String reshapeOp(long[] proposedShape, Vertex inputVertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        String variableName = lookup.get(inputVertex.getId()).getName();
        return variableName + ".reshape(" + toJavaArrayCreation(proposedShape) + ")";
    }

    private static String takeDoubleOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        TakeVertex takeVertex = (TakeVertex) vertex;
        return takeOp(takeVertex.getIndex(), takeVertex.getInputVertex(), lookup);
    }

    private static String takeIntegerOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        IntegerTakeVertex takeVertex = (IntegerTakeVertex) vertex;
        return takeOp(takeVertex.getIndex(), takeVertex.getInputVertex(), lookup);
    }

    private static String takeBooleanOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        BooleanTakeVertex takeVertex = (BooleanTakeVertex) vertex;
        return takeOp(takeVertex.getIndex(), takeVertex.getInputVertex(), lookup);
    }

    private static String takeGenericOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        GenericTakeVertex takeVertex = (GenericTakeVertex) vertex;
        return takeOp(takeVertex.getIndex(), takeVertex.getInputVertex(), lookup);
    }

    private static String takeOp(long[] index, Vertex inputVertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        String variableName = lookup.get(inputVertex.getId()).getName();
        return variableName + ".take(" + toJavaArrayCreation(index) + ");";
    }

    private static String broadcastIntegerOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        IntegerBroadcastVertex broadcastVertex = (IntegerBroadcastVertex) vertex;
        return broadcastOp(broadcastVertex.getToShape(), broadcastVertex.getInputVertex(), lookup);
    }

    private static String broadcastOp(long[] toShape, Vertex inputVertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        String variableName = lookup.get(inputVertex.getId()).getName();
        return variableName + ".broadcast(" + toJavaArrayCreation(toShape) + ");";
    }

    private static String notOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        VertexBinaryOp<?, ?> binaryOpVertex = (VertexBinaryOp<?, ?>) vertex;
        Vertex<?> left = binaryOpVertex.getLeft();
        Vertex<?> right = binaryOpVertex.getRight();

        KeanuCompiledVariable leftVariable = lookup.get(left.getReference());
        KeanuCompiledVariable rightVariable = lookup.get(right.getReference());

        return leftVariable.getName() + ".elementwiseEquals(" + rightVariable.getName() + ").not()";
    }

    private static String toJavaArrayCreation(long[] array) {
        return "new long[]{" + Arrays.stream(array).mapToObj(Long::toString).collect(Collectors.joining(",")) + "}";
    }

    private static String sliceDoubleOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        SliceVertex sliceVertex = (SliceVertex) vertex;
        return sliceOp(sliceVertex.getDimension(), sliceVertex.getIndex(), sliceVertex.getInputVertex(), lookup);
    }

    private static String sliceIntegerOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        IntegerSliceVertex sliceVertex = (IntegerSliceVertex) vertex;
        return sliceOp(sliceVertex.getDimension(), sliceVertex.getIndex(), sliceVertex.getInputVertex(), lookup);
    }

    private static String sliceBooleanOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        BooleanSliceVertex sliceVertex = (BooleanSliceVertex) vertex;
        return sliceOp(sliceVertex.getDimension(), sliceVertex.getIndex(), sliceVertex.getInputVertex(), lookup);
    }

    private static String sliceGenericOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        GenericSliceVertex sliceVertex = (GenericSliceVertex) vertex;
        return sliceOp(sliceVertex.getDimension(), sliceVertex.getIndex(), sliceVertex.getInputVertex(), lookup);
    }

    private static String sliceOp(int dimension, long index, Vertex inputVertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        String variableName = lookup.get(inputVertex.getId()).getName();
        return variableName + ".slice(" + dimension + "," + index + ")";
    }

    private static String permuteDoubleOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        PermuteVertex permuteVertex = (PermuteVertex) vertex;
        return permuteOp(permuteVertex.getRearrange(), permuteVertex.getInputVertex(), lookup);
    }

    private static String permuteIntegerOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        IntegerPermuteVertex permuteVertex = (IntegerPermuteVertex) vertex;
        return permuteOp(permuteVertex.getRearrange(), permuteVertex.getInputVertex(), lookup);
    }

    private static String permuteOp(int[] proposedShape, Vertex inputVertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        String variableName = lookup.get(inputVertex.getId()).getName();
        return variableName + ".permute(" + toJavaArrayCreation(proposedShape) + ")";
    }

    private static String toJavaArrayCreation(int[] array) {
        return "new int[]{" + Arrays.stream(array).mapToObj(Long::toString).collect(Collectors.joining(",")) + "}";
    }

    private static String concatDoubleOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        ConcatenationVertex concatenationVertex = (ConcatenationVertex) vertex;
        DoubleVertex[] operands = concatenationVertex.getOperands();
        int dimension = concatenationVertex.getDimension();

        return concatOp(dimension, operands, "DoubleTensor.concat", lookup);
    }

    private static String concatIntegerOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        IntegerConcatenationVertex concatenationVertex = (IntegerConcatenationVertex) vertex;
        IntegerVertex[] operands = concatenationVertex.getOperands();
        int dimension = concatenationVertex.getDimension();

        return concatOp(dimension, operands, "IntegerTensor.concat", lookup);
    }

    private static String concatBoolOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        BooleanConcatenationVertex concatenationVertex = (BooleanConcatenationVertex) vertex;
        BooleanVertex[] operands = concatenationVertex.getOperands();
        int dimension = concatenationVertex.getDimension();

        return concatOp(dimension, operands, "BooleanTensor.concat", lookup);
    }

    private static String concatOp(int dimension,
                                   Vertex[] operands,
                                   String concatOp,
                                   Map<VariableReference, KeanuCompiledVariable> lookup) {

        String operandArg = Arrays.stream(operands)
            .map(v -> lookup.get(v.getId()).getName())
            .collect(Collectors.joining(","));

        return concatOp + "(" + dimension + "," + operandArg + ")";

    }

    private static String sumDoubleOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        SumVertex sumVertex = (SumVertex) vertex;

        int[] dimensions = sumVertex.getOverDimensions();

        return sumOp(
            sumVertex.getInputVertex().getReference(),
            dimensions,
            lookup
        );
    }

    private static String sumIntegerOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        IntegerSumVertex sumVertex = (IntegerSumVertex) vertex;

        int[] dimensions = sumVertex.getOverDimensions();

        return sumOp(
            sumVertex.getInputVertex().getReference(),
            dimensions,
            lookup
        );
    }

    private static String sumOp(VariableReference inputReference,
                                int[] dimensions,
                                Map<VariableReference, KeanuCompiledVariable> lookup) {

        String declaration = lookup.get(inputReference).getName();

        if (dimensions != null) {
            String dims = Arrays.stream(dimensions)
                .mapToObj(i -> i + "")
                .collect(Collectors.joining(","));

            String args = "new int[]{" + dims + "}";
            return declaration + ".sum(" + args + ")";
        } else {
            return declaration + ".sum()";
        }
    }

    private static String doubleIfOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        DoubleIfVertex ifVertex = (DoubleIfVertex) vertex;

        BooleanVertex predicate = ifVertex.getPredicate();
        DoubleVertex thn = ifVertex.getThn();
        DoubleVertex els = ifVertex.getEls();

        return ifOp(predicate, thn, els, "doubleWhere", lookup);
    }

    private static String integerIfOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        IntegerIfVertex ifVertex = (IntegerIfVertex) vertex;

        BooleanVertex predicate = ifVertex.getPredicate();
        IntegerVertex thn = ifVertex.getThn();
        IntegerVertex els = ifVertex.getEls();

        return ifOp(predicate, thn, els, "integerWhere", lookup);
    }

    private static String booleanIfOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        BooleanIfVertex ifVertex = (BooleanIfVertex) vertex;

        BooleanVertex predicate = ifVertex.getPredicate();
        BooleanVertex thn = ifVertex.getThn();
        BooleanVertex els = ifVertex.getEls();

        return ifOp(predicate, thn, els, "booleanWhere", lookup);
    }

    private static String genericIfOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        IfVertex ifVertex = (IfVertex) vertex;

        BooleanVertex predicate = ifVertex.getPredicate();
        Vertex<? extends Tensor> thn = ifVertex.getThn();
        Vertex<? extends Tensor> els = ifVertex.getEls();

        return ifOp(predicate, thn, els, "where", lookup);
    }

    private static String ifOp(Vertex predicate,
                               Vertex thn,
                               Vertex els,
                               String methodName,
                               Map<VariableReference, KeanuCompiledVariable> lookup) {

        String predicateName = lookup.get(predicate.getId()).getName();
        String thnName = lookup.get(thn.getId()).getName();
        String elsName = lookup.get(els.getId()).getName();

        return predicateName + "." + methodName + "(" + thnName + "," + elsName + ")";
    }

    private static String doubleProxyOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        DoubleProxyVertex proxyVertex = (DoubleProxyVertex) vertex;
        return lookup.get(proxyVertex.getParent().getId()).getName();
    }

    private static String integerProxyOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        IntegerProxyVertex proxyVertex = (IntegerProxyVertex) vertex;
        return lookup.get(proxyVertex.getParent().getId()).getName();
    }

    private static String booleanProxyOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        BooleanProxyVertex proxyVertex = (BooleanProxyVertex) vertex;
        return lookup.get(proxyVertex.getParent().getId()).getName();
    }

    private static String numericalEqualsOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        NumericalEqualsVertex numericalEquals = (NumericalEqualsVertex) vertex;
        Vertex a = numericalEquals.getA();
        Vertex b = numericalEquals.getB();
        Number epsilon = numericalEquals.getEpsilon();

        String aName = lookup.get(a.getId()).getName();
        String bName = lookup.get(b.getId()).getName();

        return aName + ".equalsWithinEpsilon(" + bName + "," + epsilon.toString() + ")";
    }

    private static String assertOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        AssertVertex assertVertex = (AssertVertex) vertex;

        KeanuCompiledVariable predicateVariable = lookup.get(assertVertex.getPredicate().getId());

        return AssertVertex.class.getCanonicalName() + ".assertion(" +
            predicateVariable.getName() + ",\"" +
            escapeChars(assertVertex.getErrorMessage()) + "\"," +
            (assertVertex.getLabel() != null ? "\"" + escapeChars(assertVertex.getLabel().getQualifiedName()) + "\"" : "null") + ")";
    }

    private static String printOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        PrintVertex printVertex = (PrintVertex) vertex;
        KeanuCompiledVariable parentVariable = lookup.get(printVertex.getParent().getId());

        return PrintVertex.class.getCanonicalName() + ".print(" + parentVariable.getName() + ",\"" + escapeChars(printVertex.getMessage()) + "\"," + printVertex.getPrintData() + ")";
    }

    private static String escapeChars(String s) {
        return s
            .replace("\\", "\\\\")
            .replace("\t", "\\t")
            .replace("\b", "\\b")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\f", "\\f")
            .replace("\'", "\\'")
            .replace("\"", "\\\"");
    }

    private static String multiplexerOp(Vertex<?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        MultiplexerVertex muxVertex = (MultiplexerVertex) vertex;

        KeanuCompiledVariable select = lookup.get(muxVertex.getSelectorControlVertex().getId());

        String outputs = Arrays
            .stream(muxVertex.getSelectVertices())
            .map(v -> lookup.get(v.getId()).getName())
            .collect(Collectors.joining(","));

        return MultiplexerVertex.class.getCanonicalName() + ".mux(" + select.getName() + "," + outputs + ")";
    }
}
