package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.Infinitesimal;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class GaussianVertex extends ProbabilisticDoubleVector {

    private final DoubleTensorVertex mu;
    private final DoubleTensorVertex sigma;
    private final Random random;

    public GaussianVertex(DoubleTensorVertex mu, DoubleTensorVertex sigma, Random random) {
        if (mu.getValue().hasSameShapeAs(sigma.getValue())) {
            throw new IllegalArgumentException("mu and sigma must match shape");
        }

        this.mu = mu;
        this.sigma = sigma;
        this.random = random;
        setValue(sample());
        setParents(mu, sigma);
    }

    public GaussianVertex(int[] shape, DoubleTensorVertex mu, DoubleTensorVertex sigma, Random random) {

    }

    @Override
    public double density() {
        return Math.exp(logDensity());
    }

    @Override
    public double logDensity() {

        DoubleTensor muValues = mu.getValue();
        DoubleTensor sigmaValues = sigma.getValue();

        DoubleTensor logPdfs = Gaussian.logPdf(muValues, sigmaValues, getValue());

        return logPdfs.sum();
    }

    @Override
    public Map<String, Double> dDensityAtValue() {

        final double density = density();
        Map<String, Double> dlnDensityAtValue = dlnDensityAtValue();
        Map<String, Double> dDensity = new HashMap<>();
        for (String vertexId : dlnDensityAtValue.keySet()) {
            dDensity.put(vertexId, dlnDensityAtValue.get(vertexId) * density);
        }

        return dDensity;
    }

    @Override
    public Map<String, Double> dlnDensityAtValue() {

        DoubleTensor values = getValue();
        DualNumber muDualNumbers = mu.getDualNumber();
        DualNumber sigmaDualNumbers = sigma.getDualNumber();

        Map<String, Double> dlnDensity = new HashMap<>();

        for (int i = 0; i < values.length; i++) {
            Gaussian.Diff dlnP = Gaussian.dlnPdf(muDualNumbers[i].getValue(), sigmaDualNumbers[i].getValue(), values[i]);

            Map<String, Double> diff = convertDualNumbersToDiff(
                    muDualNumbers[i],
                    sigmaDualNumbers[i],
                    dlnP.dPdmu,
                    dlnP.dPdsigma,
                    dlnP.dPdx
            );

            diff.forEach((id, dlnDensity_dId) ->
                    dlnDensity.put(id, dlnDensity.getOrDefault(id, 0.0) + dlnDensity_dId)
            );
        }

        return dlnDensity;
    }

    private Map<String, DoubleTensor> convertDualNumbersToDiff(DualNumber muDualNumber,
                                                               DualNumber sigmaDualNumber,
                                                               double dPdmu,
                                                               double dPdsigma,
                                                               double dPdx) {

        Infinitesimal dPdInputsFromMu = muDualNumber.getInfinitesimal().multiplyBy(dPdmu);
        Infinitesimal dPdInputsFromSigma = sigmaDualNumber.getInfinitesimal().multiplyBy(dPdsigma);
        Infinitesimal dPdInputs = dPdInputsFromMu.add(dPdInputsFromSigma);

        dPdInputs.getInfinitesimals().put(getId(), dPdx);

        return dPdInputs.getInfinitesimals();
    }

    @Override
    public DoubleTensor sample() {
        final DoubleTensor muValues = mu.getValue();
        final DoubleTensor sigmaValues = sigma.getValue();

        double[] sampleData = new double[muValues.getLength()];

        for (int i = 0; i < sampleData.length; i++) {
            sampleData[i] = Gaussian.sample(muValues[i], sigmaValues[i], random);
        }

        final DoubleTensor samples = new DoubleTensor(, muValues.getShape());

        return samples;
    }

}
