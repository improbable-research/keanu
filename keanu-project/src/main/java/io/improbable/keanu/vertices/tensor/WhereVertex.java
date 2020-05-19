package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ReverseModePartialDerivative;

import java.util.HashMap;
import java.util.Map;


public class WhereVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends VertexImpl<TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, TensorVertex<T, TENSOR, VERTEX>, Differentiable {

    private final Vertex<BooleanTensor, ?> predicate;
    private final Vertex<TENSOR, ?> thn;
    private final Vertex<TENSOR, ?> els;
    private final Class<?> type;

    protected static final String PREDICATE_NAME = "predicate";
    protected static final String THEN_NAME = "then";
    protected static final String ELSE_NAME = "else";

    @ExportVertexToPythonBindings
    public WhereVertex(@LoadVertexParam(PREDICATE_NAME) Vertex<BooleanTensor, ?> predicate,
                       @LoadVertexParam(THEN_NAME) Vertex<TENSOR, ?> thn,
                       @LoadVertexParam(ELSE_NAME) Vertex<TENSOR, ?> els) {
        super(TensorShapeValidation.checkTernaryConditionShapeIsValid(predicate.getShape(), thn.getShape(), els.getShape()));
        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        this.type = thn.ofType();
        setParents(predicate, thn, els);
    }

    @SaveVertexParam(PREDICATE_NAME)
    public Vertex<BooleanTensor, ?> getPredicate() {
        return predicate;
    }

    @SaveVertexParam(THEN_NAME)
    public Vertex<TENSOR, ?> getThn() {
        return thn;
    }

    @SaveVertexParam(ELSE_NAME)
    public Vertex<TENSOR, ?> getEls() {
        return els;
    }

    @Override
    public TENSOR calculate() {
        return thn.getValue().where(predicate.getValue(), els.getValue());
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {

        ForwardModePartialDerivative thnPartial = derivativeOfParentsWithRespectToInput.getOrDefault(thn, ForwardModePartialDerivative.EMPTY);
        ForwardModePartialDerivative elsPartial = derivativeOfParentsWithRespectToInput.getOrDefault(els, ForwardModePartialDerivative.EMPTY);
        BooleanTensor predicateValue = predicate.getValue();

        if (predicateValue.allTrue().scalar()) {
            return thnPartial;
        } else if (predicateValue.allFalse().scalar()) {
            return elsPartial;
        } else {
            return thnPartial.multiply(predicateValue.toDoubleMask())
                .add(elsPartial.multiply(predicateValue.not().toDoubleMask()), this.getShape());
        }
    }

    @Override
    public Map<Vertex, ReverseModePartialDerivative> reverseModeAutoDifferentiation(ReverseModePartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, ReverseModePartialDerivative> partials = new HashMap<>();
        BooleanTensor predicateValue = predicate.getValue();
        partials.put(thn, derivativeOfOutputWithRespectToSelf
            .multiply(predicateValue.toDoubleMask()));
        partials.put(els, derivativeOfOutputWithRespectToSelf
            .multiply(predicateValue.not().toDoubleMask()));
        return partials;
    }

    @Override
    public VERTEX wrap(NonProbabilisticVertex<TENSOR, VERTEX> vertex) {
        return null;
    }

    @Override
    public Class<?> ofType() {
        return type;
    }
}
