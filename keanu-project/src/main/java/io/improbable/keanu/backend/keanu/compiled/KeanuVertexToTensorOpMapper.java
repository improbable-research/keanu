package io.improbable.keanu.backend.keanu.compiled;

import com.google.common.primitives.Booleans;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.VertexUnaryOp;
import io.improbable.keanu.vertices.tensor.BroadcastVertex;
import io.improbable.keanu.vertices.tensor.DiagPartVertex;
import io.improbable.keanu.vertices.tensor.DiagVertex;
import io.improbable.keanu.vertices.tensor.GetBooleanIndexVertex;
import io.improbable.keanu.vertices.tensor.PermuteVertex;
import io.improbable.keanu.vertices.tensor.ReshapeVertex;
import io.improbable.keanu.vertices.tensor.SliceVertex;
import io.improbable.keanu.vertices.tensor.StridedSliceVertex;
import io.improbable.keanu.vertices.tensor.TakeVertex;
import io.improbable.keanu.vertices.tensor.WhereVertex;
import io.improbable.keanu.vertices.tensor.bool.BooleanVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.BooleanProxyVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.CastNumberToBooleanVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.CastToBooleanVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.ConstantBooleanVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.AndBinaryVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.OrBinaryVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.XorBinaryVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.compare.EqualsVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.compare.GreaterThanOrEqualVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.compare.LessThanVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.compare.NumericalEqualsVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.multiple.BooleanConcatenationVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.multiple.BooleanToDoubleMaskVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.multiple.BooleanToIntegerMaskVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.AllFalseVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.AllTrueVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.AnyFalseVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.AnyTrueVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.IsFiniteVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.IsInfiniteVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.IsNaNVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.IsNegativeInfinityVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.IsPositiveInfinityVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.NotBinaryVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary.NotNaNVertex;
import io.improbable.keanu.vertices.tensor.generic.nonprobabilistic.MultiplexerVertex;
import io.improbable.keanu.vertices.tensor.generic.nonprobabilistic.PrintVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.CastNumberToIntegerVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.IntegerProxyVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.operators.multiple.IntegerConcatenationVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.operators.unary.ArgMaxVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.operators.unary.ArgMinVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.operators.unary.NaNArgMaxVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.operators.unary.NaNArgMinVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.operators.unary.ModVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.CastNumberToDoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.binary.ArcTan2Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.binary.LogAddExp2Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.binary.LogAddExpVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.binary.SafeLogTimesVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcCosVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcCoshVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcSinVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcSinhVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcTanVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcTanhVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.CeilVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.CholeskyDecompositionVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.CosVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.CoshVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.DigammaVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.Exp2Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ExpM1Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ExpVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.FloorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.Log10Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.Log1pVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.Log2Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.LogGammaVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.LogVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.MatrixDeterminantVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.MatrixInverseVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.MeanVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ReplaceNaNVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.RoundVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.SigmoidVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.SinVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.SinhVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.StandardDeviationVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.StandardizeVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.TanVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.TanhVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.TrigammaVertex;
import io.improbable.keanu.vertices.tensor.number.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.tensor.number.operators.binary.DifferenceVertex;
import io.improbable.keanu.vertices.tensor.number.operators.binary.DivisionVertex;
import io.improbable.keanu.vertices.tensor.number.operators.binary.GreaterThanMaskVertex;
import io.improbable.keanu.vertices.tensor.number.operators.binary.GreaterThanOrEqualToMaskVertex;
import io.improbable.keanu.vertices.tensor.number.operators.binary.LessThanMaskVertex;
import io.improbable.keanu.vertices.tensor.number.operators.binary.LessThanOrEqualToMaskVertex;
import io.improbable.keanu.vertices.tensor.number.operators.binary.MatrixMultiplicationVertex;
import io.improbable.keanu.vertices.tensor.number.operators.binary.MaxVertex;
import io.improbable.keanu.vertices.tensor.number.operators.binary.MinVertex;
import io.improbable.keanu.vertices.tensor.number.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.tensor.number.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.tensor.number.operators.binary.TensorMultiplicationVertex;
import io.improbable.keanu.vertices.tensor.number.operators.ternary.SetWithMaskVertex;
import io.improbable.keanu.vertices.tensor.number.operators.unary.AbsVertex;
import io.improbable.keanu.vertices.tensor.number.operators.unary.CumProdVertex;
import io.improbable.keanu.vertices.tensor.number.operators.unary.CumSumVertex;
import io.improbable.keanu.vertices.tensor.number.operators.unary.MaxUnaryVertex;
import io.improbable.keanu.vertices.tensor.number.operators.unary.MinUnaryVertex;
import io.improbable.keanu.vertices.tensor.number.operators.unary.ProductVertex;
import io.improbable.keanu.vertices.tensor.number.operators.unary.SumVertex;
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

        //Tensor ops
        opMappers.put(ReshapeVertex.class, KeanuVertexToTensorOpMapper::reshapeOp);
        opMappers.put(PermuteVertex.class, KeanuVertexToTensorOpMapper::permuteOp);
        opMappers.put(BroadcastVertex.class, KeanuVertexToTensorOpMapper::broadcastOp);
        opMappers.put(TakeVertex.class, KeanuVertexToTensorOpMapper::takeOp);
        opMappers.put(SliceVertex.class, KeanuVertexToTensorOpMapper::sliceOp);
        opMappers.put(StridedSliceVertex.class, KeanuVertexToTensorOpMapper::stridedSliceOp);
        opMappers.put(DiagVertex.class, fluentUnaryOp("diag"));
        opMappers.put(DiagPartVertex.class, fluentUnaryOp("diagPart"));
        opMappers.put(GetBooleanIndexVertex.class, fluentBinaryOp("get"));
        opMappers.put(WhereVertex.class, KeanuVertexToTensorOpMapper::genericIfOp);

        //Number ops
        opMappers.put(DifferenceVertex.class, fluentBinaryOp("minus", "minusInPlace"));
        opMappers.put(AdditionVertex.class, fluentBinaryOp("plus", "plusInPlace"));
        opMappers.put(MultiplicationVertex.class, fluentBinaryOp("times", "timesInPlace"));
        opMappers.put(DivisionVertex.class, fluentBinaryOp("div", "divInPlace"));
        opMappers.put(SumVertex.class, KeanuVertexToTensorOpMapper::sumOp);
        opMappers.put(ArgMinVertex.class, KeanuVertexToTensorOpMapper::argMinOp);
        opMappers.put(ArgMaxVertex.class, KeanuVertexToTensorOpMapper::argMaxOp);
        opMappers.put(NaNArgMaxVertex.class, KeanuVertexToTensorOpMapper::nanArgMaxOp);
        opMappers.put(NaNArgMinVertex.class, KeanuVertexToTensorOpMapper::nanArgMinOp);
        opMappers.put(CumSumVertex.class, KeanuVertexToTensorOpMapper::cumSumOp);
        opMappers.put(CumProdVertex.class, KeanuVertexToTensorOpMapper::cumProdOp);
        opMappers.put(ProductVertex.class, KeanuVertexToTensorOpMapper::productOp);
        opMappers.put(MatrixMultiplicationVertex.class, fluentBinaryOp("matrixMultiply"));
        opMappers.put(TensorMultiplicationVertex.class, KeanuVertexToTensorOpMapper::tensorMultiply);
        opMappers.put(PowerVertex.class, fluentBinaryOp("pow", "powInPlace"));
        opMappers.put(AbsVertex.class, fluentUnaryOp("abs", "absInPlace"));

        opMappers.put(GreaterThanOrEqualToMaskVertex.class, fluentBinaryOp("greaterThanOrEqualToMask"));
        opMappers.put(GreaterThanMaskVertex.class, fluentBinaryOp("greaterThanMask"));
        opMappers.put(LessThanOrEqualToMaskVertex.class, fluentBinaryOp("lessThanOrEqualToMask"));
        opMappers.put(LessThanMaskVertex.class, fluentBinaryOp("lessThanMask"));
        opMappers.put(SetWithMaskVertex.class, KeanuVertexToTensorOpMapper::setWithMaskOp);
        opMappers.put(MaxVertex.class, fluentBinaryOp("max"));
        opMappers.put(MinVertex.class, fluentBinaryOp("min"));
        opMappers.put(MaxUnaryVertex.class, fluentUnaryOp("max"));
        opMappers.put(MinUnaryVertex.class, fluentUnaryOp("min"));

        //Floating point ops
        opMappers.put(CosVertex.class, fluentUnaryOp("cos", "cosInPlace"));
        opMappers.put(CoshVertex.class, fluentUnaryOp("cosh", "coshInPlace"));
        opMappers.put(ArcCosVertex.class, fluentUnaryOp("acos", "acosInPlace"));
        opMappers.put(TanhVertex.class, fluentUnaryOp("tanh", "tanhInPlace"));
        opMappers.put(ArcTanhVertex.class, fluentUnaryOp("atanh", "atanhInPlace"));
        opMappers.put(ArcCoshVertex.class, fluentUnaryOp("acosh", "acoshInPlace"));
        opMappers.put(SinhVertex.class, fluentUnaryOp("sinh", "sinhInPlace"));
        opMappers.put(ArcSinhVertex.class, fluentUnaryOp("asinh", "asinhInPlace"));
        opMappers.put(ArcTan2Vertex.class, fluentBinaryOp("atan2", "atan2InPlace"));
        opMappers.put(SinVertex.class, fluentUnaryOp("sin", "sinInPlace"));
        opMappers.put(TanVertex.class, fluentUnaryOp("tan", "tanInPlace"));
        opMappers.put(ArcSinVertex.class, fluentUnaryOp("asin", "asinInPlace"));
        opMappers.put(ArcTanVertex.class, fluentUnaryOp("atan", "atanInPlace"));

        opMappers.put(CeilVertex.class, fluentUnaryOp("ceil", "ceilInPlace"));
        opMappers.put(FloorVertex.class, fluentUnaryOp("floor", "floorInPlace"));
        opMappers.put(RoundVertex.class, fluentUnaryOp("round", "roundInPlace"));

        opMappers.put(ExpVertex.class, fluentUnaryOp("exp", "expInPlace"));
        opMappers.put(LogVertex.class, fluentUnaryOp("log", "logInPlace"));
        opMappers.put(LogGammaVertex.class, fluentUnaryOp("logGamma", "logGammaInPlace"));
        opMappers.put(Log2Vertex.class, fluentUnaryOp("log2", "log2InPlace"));
        opMappers.put(Exp2Vertex.class, fluentUnaryOp("exp2", "exp2InPlace"));
        opMappers.put(Log10Vertex.class, fluentUnaryOp("log10", "log10InPlace"));
        opMappers.put(LogAddExp2Vertex.class, fluentBinaryOp("logAddExp2"));
        opMappers.put(LogAddExpVertex.class, fluentBinaryOp("logAddExp2"));
        opMappers.put(Log1pVertex.class, fluentUnaryOp("log1p", "log1pInPlace"));
        opMappers.put(ExpM1Vertex.class, fluentUnaryOp("expM1", "expM1InPlace"));

        opMappers.put(DigammaVertex.class, fluentUnaryOp("digamma", "digammaInPlace"));
        opMappers.put(TrigammaVertex.class, fluentUnaryOp("trigamma", "trigammaInPlace"));
        opMappers.put(SigmoidVertex.class, fluentUnaryOp("sigmoid", "sigmoidInPlace"));
        opMappers.put(MatrixDeterminantVertex.class, fluentUnaryOp("matrixDeterminant"));
        opMappers.put(MatrixInverseVertex.class, fluentUnaryOp("matrixInverse"));
        opMappers.put(CholeskyDecompositionVertex.class, fluentUnaryOp("choleskyDecomposition"));
        opMappers.put(StandardizeVertex.class, fluentUnaryOp("standardize"));
        opMappers.put(MeanVertex.class, KeanuVertexToTensorOpMapper::meanOp);
        opMappers.put(StandardDeviationVertex.class, fluentUnaryOp("standardDeviation"));
        opMappers.put(ReplaceNaNVertex.class, KeanuVertexToTensorOpMapper::replaceNaNOp);
        opMappers.put(NotNaNVertex.class, fluentUnaryOp("notNaN"));
        opMappers.put(IsNaNVertex.class, fluentUnaryOp("isNaN"));
        opMappers.put(IsNegativeInfinityVertex.class, fluentUnaryOp("isNegativeInfinity"));
        opMappers.put(IsPositiveInfinityVertex.class, fluentUnaryOp("isPositiveInfinity"));
        opMappers.put(IsInfiniteVertex.class, fluentUnaryOp("isInfinite"));
        opMappers.put(IsFiniteVertex.class, fluentUnaryOp("isFinite"));
        opMappers.put(SafeLogTimesVertex.class, fluentBinaryOp("safeLogTimes", "safeLogTimesInPlace"));

        //Fixed point ops
        opMappers.put(ModVertex.class, fluentBinaryOp("mod", "modInPlace"));

        //Double ops
        opMappers.put(ConcatenationVertex.class, KeanuVertexToTensorOpMapper::concatDoubleOp);
        opMappers.put(CastNumberToDoubleVertex.class, fluentUnaryOp("toDouble"));
        opMappers.put(DoubleProxyVertex.class, KeanuVertexToTensorOpMapper::doubleProxyOp);

        //Integer ops
        opMappers.put(IntegerConcatenationVertex.class, KeanuVertexToTensorOpMapper::concatIntegerOp);
        opMappers.put(CastNumberToIntegerVertex.class, fluentUnaryOp("toInteger"));
        opMappers.put(IntegerProxyVertex.class, KeanuVertexToTensorOpMapper::integerProxyOp);

        //Booleans ops
        opMappers.put(BooleanConcatenationVertex.class, KeanuVertexToTensorOpMapper::concatBoolOp);
        opMappers.put(GreaterThanOrEqualVertex.class, fluentBinaryOp("greaterThanOrEqual"));
        opMappers.put(GreaterThanVertex.class, fluentBinaryOp("greaterThan"));
        opMappers.put(LessThanOrEqualVertex.class, fluentBinaryOp("lessThanOrEqual"));
        opMappers.put(LessThanVertex.class, fluentBinaryOp("lessThan"));
        opMappers.put(EqualsVertex.class, fluentBinaryOp("elementwiseEquals"));
        opMappers.put(NotEqualsVertex.class, KeanuVertexToTensorOpMapper::notOp);
        opMappers.put(NumericalEqualsVertex.class, KeanuVertexToTensorOpMapper::numericalEqualsOp);
        opMappers.put(OrBinaryVertex.class, fluentBinaryOp("or", "orInPlace"));
        opMappers.put(AndBinaryVertex.class, fluentBinaryOp("and", "andInPlace"));
        opMappers.put(NotBinaryVertex.class, fluentUnaryOp("not", "notInPlace"));
        opMappers.put(XorBinaryVertex.class, fluentUnaryOp("xor", "xorInPlace"));
        opMappers.put(CastToBooleanVertex.class, fluentUnaryOp("toBoolean"));
        opMappers.put(CastNumberToBooleanVertex.class, fluentUnaryOp("toBoolean"));
        opMappers.put(BooleanToIntegerMaskVertex.class, fluentUnaryOp("toIntegerMask"));
        opMappers.put(BooleanToDoubleMaskVertex.class, fluentUnaryOp("toDoubleMask"));
        opMappers.put(AllTrueVertex.class, fluentUnaryOp("allTrue"));
        opMappers.put(AnyTrueVertex.class, fluentUnaryOp("anyTrue"));
        opMappers.put(AllFalseVertex.class, fluentUnaryOp("allFalse"));
        opMappers.put(AnyFalseVertex.class, fluentUnaryOp("anyFalse"));
        opMappers.put(BooleanProxyVertex.class, KeanuVertexToTensorOpMapper::booleanProxyOp);

        //Constants
        opMappers.put(ConstantIntegerVertex.class, KeanuVertexToTensorOpMapper::constant);
        opMappers.put(ConstantDoubleVertex.class, KeanuVertexToTensorOpMapper::constant);
        opMappers.put(ConstantBooleanVertex.class, KeanuVertexToTensorOpMapper::constant);

        //Generic ops
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
        String apply(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup);
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
            Vertex<?, ?> left = binaryOpVertex.getLeft();
            Vertex<?, ?> right = binaryOpVertex.getRight();

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
            Vertex<?, ?> input = unaryOpVertex.getInputVertex();

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
    private static boolean isLastChildByTopographicalSort(Vertex<?, ?> child, Vertex<?, ?> parent) {
        Optional<Vertex> last = parent.getChildren().stream()
            .max(Comparator.comparing(Vertex::getId));

        return last
            .map(v -> v.getId().equals(child.getId()))
            .orElse(false);
    }

    private static String constant(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        throw new IllegalArgumentException("Constant should not be operation mapped");
    }

    private static String tensorMultiply(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        TensorMultiplicationVertex tmul = (TensorMultiplicationVertex) vertex;
        String leftName = lookup.get(tmul.getLeft().getId()).getName();
        String rightName = lookup.get(tmul.getRight().getId()).getName();
        int[] dimsLeft = tmul.getDimsLeft();
        int[] dimsRight = tmul.getDimsRight();
        return leftName + ".tensorMultiply(" + rightName + "," + toJavaArrayCreation(dimsLeft) + "," + toJavaArrayCreation(dimsRight) + ");";
    }

    private static String setWithMaskOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        SetWithMaskVertex setWithMaskVertex = (SetWithMaskVertex) vertex;
        Vertex mask = setWithMaskVertex.getMask();
        Vertex operand = setWithMaskVertex.getOperand();
        Vertex setValue = setWithMaskVertex.getSetValue();

        String operandName = lookup.get(operand.getId()).getName();
        String maskName = lookup.get(mask.getId()).getName();
        String setValueName = lookup.get(setValue.getId()).getName();

        return operandName + ".setWithMask(" + maskName + "," + setValueName + ".scalar())";
    }

    private static String reshapeOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        ReshapeVertex reshapeVertex = (ReshapeVertex) vertex;
        String variableName = lookup.get(reshapeVertex.getInputVertex().getId()).getName();
        return variableName + ".reshape(" + toJavaArrayCreation(reshapeVertex.getProposedShape()) + ")";
    }

    private static String takeOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        TakeVertex takeVertex = (TakeVertex) vertex;
        String variableName = lookup.get(takeVertex.getInputVertex().getId()).getName();
        return variableName + ".take(" + toJavaArrayCreation(takeVertex.getIndex()) + ");";
    }

    private static String broadcastOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        BroadcastVertex broadcastVertex = (BroadcastVertex) vertex;
        String variableName = lookup.get(broadcastVertex.getInputVertex().getId()).getName();
        return variableName + ".broadcast(" + toJavaArrayCreation(broadcastVertex.getToShape()) + ");";
    }

    private static String notOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        VertexBinaryOp<?, ?> binaryOpVertex = (VertexBinaryOp<?, ?>) vertex;
        Vertex<?, ?> left = binaryOpVertex.getLeft();
        Vertex<?, ?> right = binaryOpVertex.getRight();

        KeanuCompiledVariable leftVariable = lookup.get(left.getReference());
        KeanuCompiledVariable rightVariable = lookup.get(right.getReference());

        return leftVariable.getName() + ".elementwiseEquals(" + rightVariable.getName() + ").not()";
    }

    private static String sliceOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        SliceVertex sliceVertex = (SliceVertex) vertex;
        String variableName = lookup.get(sliceVertex.getInputVertex().getId()).getName();
        return variableName + ".slice(" + sliceVertex.getDimension() + "," + sliceVertex.getIndex() + ")";
    }

    private static String stridedSliceOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        StridedSliceVertex sliceVertex = (StridedSliceVertex) vertex;
        String variableName = lookup.get(sliceVertex.getInputVertex().getId()).getName();

        String slicerClassName = Slicer.class.getCanonicalName();
        return variableName + ".slice(new " + slicerClassName + "(" +
            toJavaArrayCreation(sliceVertex.getStart()) + "," +
            toJavaArrayCreation(sliceVertex.getEnd()) + "," +
            toJavaArrayCreation(sliceVertex.getStride()) + "," +
            sliceVertex.getEllipsis() + "," +
            toJavaArrayCreation(sliceVertex.getUpperBoundStop()) + "," +
            toJavaArrayCreation(sliceVertex.getDropDimension()) +
            "))";
    }

    private static String permuteOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        PermuteVertex permuteVertex = (PermuteVertex) vertex;
        String variableName = lookup.get(permuteVertex.getId()).getName();
        return variableName + ".permute(" + toJavaArrayCreation(permuteVertex.getRearrange()) + ")";
    }

    private static String concatDoubleOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        ConcatenationVertex concatenationVertex = (ConcatenationVertex) vertex;
        Vertex[] operands = concatenationVertex.getOperands();
        int dimension = concatenationVertex.getDimension();

        return concatOp(dimension, operands, "DoubleTensor.concat", lookup);
    }

    private static String concatIntegerOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        IntegerConcatenationVertex concatenationVertex = (IntegerConcatenationVertex) vertex;
        IntegerVertex[] operands = concatenationVertex.getOperands();
        int dimension = concatenationVertex.getDimension();

        return concatOp(dimension, operands, "IntegerTensor.concat", lookup);
    }

    private static String concatBoolOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
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

    private static String replaceNaNOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        ReplaceNaNVertex replaceNaNVertex = (ReplaceNaNVertex) vertex;
        String variableName = lookup.get(replaceNaNVertex.getInputVertex().getId()).getName();
        return variableName + ".replaceNaN(" + replaceNaNVertex.getReplaceWithValue().doubleValue() + ");";
    }

    private static String cumSumOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        CumSumVertex cumSumVertex = (CumSumVertex) vertex;

        int dim = cumSumVertex.getRequestedDimension();
        String declaration = lookup.get(cumSumVertex.getInputVertex().getReference()).getName();

        return declaration + ".cumSum(" + dim + ")";
    }

    private static String cumProdOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        CumProdVertex cumProdVertex = (CumProdVertex) vertex;

        int dim = cumProdVertex.getRequestedDimension();
        String declaration = lookup.get(cumProdVertex.getInputVertex().getReference()).getName();

        return declaration + ".cumProd(" + dim + ")";
    }

    private static String argMinOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        ArgMinVertex opVertex = (ArgMinVertex) vertex;

        Integer dim = opVertex.getAxis();
        String declaration = lookup.get(opVertex.getInputVertex().getReference()).getName();

        if (dim != null) {
            return declaration + ".argMin(" + dim + ")";
        } else {
            return declaration + ".argMin()";
        }
    }

    private static String argMaxOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        ArgMaxVertex opVertex = (ArgMaxVertex) vertex;

        Integer dim = opVertex.getAxis();
        String declaration = lookup.get(opVertex.getInputVertex().getReference()).getName();

        if (dim != null) {
            return declaration + ".argMax(" + dim + ")";
        } else {
            return declaration + ".argMax()";
        }
    }

    private static String nanArgMaxOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        NaNArgMaxVertex opVertex = (NaNArgMaxVertex) vertex;

        Integer dim = opVertex.getAxis();
        String declaration = lookup.get(opVertex.getInputVertex().getReference()).getName();

        if (dim != null) {
            return declaration + ".nanArgMax(" + dim + ")";
        } else {
            return declaration + ".nanArgMax()";
        }
    }

    private static String nanArgMinOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        NaNArgMaxVertex opVertex = (NaNArgMaxVertex) vertex;

        Integer dim = opVertex.getAxis();
        String declaration = lookup.get(opVertex.getInputVertex().getReference()).getName();

        if (dim != null) {
            return declaration + ".nanArgMin(" + dim + ")";
        } else {
            return declaration + ".nanArgMin()";
        }
    }

    private static String sumOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        SumVertex sumVertex = (SumVertex) vertex;

        int[] dimensions = sumVertex.getOverDimensions();
        String declaration = lookup.get(sumVertex.getInputVertex().getReference()).getName();

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

    private static String meanOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        MeanVertex meanVertex = (MeanVertex) vertex;

        int[] dimensions = meanVertex.getOverDimensions();
        String declaration = lookup.get(meanVertex.getInputVertex().getReference()).getName();

        if (dimensions != null) {
            String dims = Arrays.stream(dimensions)
                .mapToObj(i -> i + "")
                .collect(Collectors.joining(","));

            String args = "new int[]{" + dims + "}";
            return declaration + ".mean(" + args + ")";
        } else {
            return declaration + ".mean()";
        }
    }

    private static String productOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        ProductVertex productVertex = (ProductVertex) vertex;

        int[] dimensions = productVertex.getOverDimensions();
        String declaration = lookup.get(productVertex.getInputVertex().getReference()).getName();

        if (dimensions != null) {
            String dims = Arrays.stream(dimensions)
                .mapToObj(i -> i + "")
                .collect(Collectors.joining(","));

            String args = "new int[]{" + dims + "}";
            return declaration + ".product(" + args + ")";
        } else {
            return declaration + ".product()";
        }
    }

    private static String genericIfOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        WhereVertex whereVertex = (WhereVertex) vertex;

        Vertex<BooleanTensor, ?> predicate = whereVertex.getPredicate();
        Vertex<?, ?> thn = whereVertex.getThn();
        Vertex<?, ?> els = whereVertex.getEls();

        String predicateName = lookup.get(predicate.getId()).getName();
        String thnName = lookup.get(thn.getId()).getName();
        String elsName = lookup.get(els.getId()).getName();

        return thnName + ".where(" + predicateName + "," + elsName + ")";
    }

    private static String doubleProxyOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        DoubleProxyVertex proxyVertex = (DoubleProxyVertex) vertex;
        return lookup.get(proxyVertex.getParent().getId()).getName();
    }

    private static String integerProxyOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        IntegerProxyVertex proxyVertex = (IntegerProxyVertex) vertex;
        return lookup.get(proxyVertex.getParent().getId()).getName();
    }

    private static String booleanProxyOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        BooleanProxyVertex proxyVertex = (BooleanProxyVertex) vertex;
        return lookup.get(proxyVertex.getParent().getId()).getName();
    }

    private static String numericalEqualsOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        NumericalEqualsVertex numericalEquals = (NumericalEqualsVertex) vertex;
        Vertex a = numericalEquals.getA();
        Vertex b = numericalEquals.getB();
        Number epsilon = numericalEquals.getEpsilon();

        String aName = lookup.get(a.getId()).getName();
        String bName = lookup.get(b.getId()).getName();

        return aName + ".equalsWithinEpsilon(" + bName + "," + epsilon.toString() + ")";
    }

    private static String assertOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        AssertVertex assertVertex = (AssertVertex) vertex;

        KeanuCompiledVariable predicateVariable = lookup.get(assertVertex.getPredicate().getId());

        return AssertVertex.class.getCanonicalName() + ".assertion(" +
            predicateVariable.getName() + ",\"" +
            escapeChars(assertVertex.getErrorMessage()) + "\"," +
            (assertVertex.getLabel() != null ? "\"" + escapeChars(assertVertex.getLabel().getQualifiedName()) + "\"" : "null") + ")";
    }

    private static String printOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
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

    private static String multiplexerOp(Vertex<?, ?> vertex, Map<VariableReference, KeanuCompiledVariable> lookup) {
        MultiplexerVertex muxVertex = (MultiplexerVertex) vertex;

        KeanuCompiledVariable select = lookup.get(muxVertex.getSelectorControlVertex().getId());

        String outputs = Arrays
            .stream(muxVertex.getSelectVertices())
            .map(v -> lookup.get(v.getId()).getName())
            .collect(Collectors.joining(","));

        return MultiplexerVertex.class.getCanonicalName() + ".mux(" + select.getName() + "," + outputs + ")";
    }

    private static String toJavaArrayCreation(long[] array) {
        return "new long[]{" + Longs.join(",", array) + "}";
    }

    private static String toJavaArrayCreation(int[] array) {
        return "new int[]{" + Ints.join(",", array) + "}";
    }

    private static String toJavaArrayCreation(boolean[] array) {
        return "new boolean[]{" + Booleans.join(",", array) + "}";
    }
}
