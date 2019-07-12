package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;


public class DoubleIfVertex extends VertexImpl<DoubleTensor, DoubleVertex> implements DoubleVertex, Differentiable, NonProbabilistic<DoubleTensor> {

    private final Vertex<BooleanTensor, ?> predicate;
    private final Vertex<DoubleTensor, ?> thn;
    private final Vertex<DoubleTensor, ?> els;
    protected static final String PREDICATE_NAME = "predicate";
    protected static final String THEN_NAME = "then";
    protected static final String ELSE_NAME = "else";

    @ExportVertexToPythonBindings
    public DoubleIfVertex(@LoadVertexParam(PREDICATE_NAME) Vertex<BooleanTensor, ?> predicate,
                          @LoadVertexParam(THEN_NAME) Vertex<DoubleTensor, ?> thn,
                          @LoadVertexParam(ELSE_NAME) Vertex<DoubleTensor, ?> els) {
        super(TensorShapeValidation.checkTernaryConditionShapeIsValid(predicate.getShape(), thn.getShape(), els.getShape()));
        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        setParents(predicate, thn, els);
    }

    @SaveVertexParam(PREDICATE_NAME)
    public Vertex<BooleanTensor, ?> getPredicate() {
        return predicate;
    }

    @SaveVertexParam(THEN_NAME)
    public Vertex<DoubleTensor, ?> getThn() {
        return thn;
    }

    @SaveVertexParam(ELSE_NAME)
    public Vertex<DoubleTensor, ?> getEls() {
        return els;
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {

        PartialDerivative thnPartial = derivativeOfParentsWithRespectToInput.getOrDefault(thn, PartialDerivative.EMPTY);
        PartialDerivative elsPartial = derivativeOfParentsWithRespectToInput.getOrDefault(els, PartialDerivative.EMPTY);
        BooleanTensor predicateValue = predicate.getValue();

        if (predicateValue.allTrue().scalar()) {
            return thnPartial;
        } else if (predicateValue.allFalse().scalar()) {
            return elsPartial;
        } else {
            return thnPartial.multiplyAlongOfDimensions(predicateValue.toDoubleMask())
                .add(elsPartial.multiplyAlongOfDimensions(predicateValue.not().toDoubleMask()));
        }
    }

    @Override
    public DoubleTensor calculate() {
        return op(predicate.getValue(), thn.getValue(), els.getValue());
    }

    private DoubleTensor op(BooleanTensor predicate, DoubleTensor thn, DoubleTensor els) {
        return predicate.doubleWhere(thn, els);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        BooleanTensor predicateValue = predicate.getValue();
        partials.put(thn, derivativeOfOutputWithRespectToSelf
            .multiplyAlongWrtDimensions(predicateValue.toDoubleMask()));
        partials.put(els, derivativeOfOutputWithRespectToSelf
            .multiplyAlongWrtDimensions(predicateValue.not().toDoubleMask()));
        return partials;
    }
}
