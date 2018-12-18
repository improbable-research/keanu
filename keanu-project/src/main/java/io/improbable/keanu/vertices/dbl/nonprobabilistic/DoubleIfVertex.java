package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;


public class DoubleIfVertex extends DoubleVertex implements Differentiable, NonProbabilistic<DoubleTensor> {

    private final Vertex<? extends BooleanTensor> predicate;
    private final DoubleVertex thn;
    private final DoubleVertex els;
    protected static final String PREDICATE_NAME = "predicate";
    protected static final String THEN_NAME = "then";
    protected static final String ELSE_NAME = "else";

    @ExportVertexToPythonBindings
    public DoubleIfVertex(@LoadVertexParam(PREDICATE_NAME) Vertex<? extends BooleanTensor> predicate,
                          @LoadVertexParam(THEN_NAME) DoubleVertex thn,
                          @LoadVertexParam(ELSE_NAME) DoubleVertex els) {
        super(TensorShapeValidation.checkTernaryConditionShapeIsValid(predicate.getShape(), thn.getShape(), els.getShape()));
        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        setParents(predicate, thn, els);
    }

    @SaveVertexParam(PREDICATE_NAME)
    public Vertex<? extends BooleanTensor> getPredicate() {
        return predicate;
    }

    @SaveVertexParam(THEN_NAME)
    public DoubleVertex getThn() {
        return thn;
    }

    @SaveVertexParam(ELSE_NAME)
    public DoubleVertex getEls() {
        return els;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op(predicate.sample(random), thn.sample(random), els.sample(random));
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {

        long[] ofShape = getShape();
        PartialDerivative thnPartial = derivativeOfParentsWithRespectToInput.getOrDefault(thn, PartialDerivative.EMPTY);
        PartialDerivative elsPartial = derivativeOfParentsWithRespectToInput.getOrDefault(els, PartialDerivative.EMPTY);
        BooleanTensor predicateValue = predicate.getValue();

        if (predicateValue.allTrue()) {
            return thnPartial;
        } else if (predicateValue.allFalse()) {
            return elsPartial;
        } else {
            return thnPartial.multiplyAlongOfDimensions(predicateValue.toDoubleMask(), ofShape)
                .add(elsPartial.multiplyAlongOfDimensions(predicateValue.not().toDoubleMask(), ofShape));
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
            .multiplyAlongWrtDimensions(predicateValue.toDoubleMask(), this.getShape()));
        partials.put(els, derivativeOfOutputWithRespectToSelf
            .multiplyAlongWrtDimensions(predicateValue.not().toDoubleMask(), this.getShape()));
        return partials;
    }

}
