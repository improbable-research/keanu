package io.improbable.keanu.e2e.weaponMaterials;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.EqualsVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleCPTVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.ConditionalProbabilityTable;
import io.improbable.keanu.vertices.generic.probabilistic.discrete.CategoricalVertex;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class WeaponMaterialsTest {

    public enum Weapon {
        POISON,
        SPANNER,
        CLUB,
        KNIFE
    }

    public enum Material {
        IRON,
        WOOD,
        STEAL,
        RUBBER,
        FUNNY_SMELL
    }

    @Test
    public void testBooleanConditions() {
        CategoricalVertex<Weapon> weapon = getWeaponPrior();
        CategoricalVertex<Material> material = getWeaponMaterialPrior();
        DoubleCPTVertex cpt = createCPTWithBooleanInputs(weapon, material);

        material.observe(GenericTensor.scalar(Material.WOOD));
        NetworkSamples samples = takeSamples(cpt, weapon, material);
        Map<Weapon, Double> weaponSampleProportion = calculateSampleProportions(samples, weapon);
        Assert.assertEquals(1.0, weaponSampleProportion.get(Weapon.CLUB), 0.001);
    }

    @Test
    public void testEnumConditions() {
        CategoricalVertex<Weapon> weapon = getWeaponPrior();
        CategoricalVertex<Material> material = getWeaponMaterialPrior();
        DoubleCPTVertex cpt = createCPTWithEnumInputs(weapon, material);

        material.observe(GenericTensor.scalar(Material.WOOD));
        NetworkSamples samples = takeSamples(cpt, weapon, material);
        Map<Weapon, Double> weaponSampleProportion = calculateSampleProportions(samples, weapon);
        Assert.assertEquals(1.0, weaponSampleProportion.get(Weapon.CLUB), 0.001);
    }

    @Test
    public void testMixedConditions() {
        CategoricalVertex<Weapon> weapon = getWeaponPrior();
        CategoricalVertex<Material> material = getWeaponMaterialPrior();
        DoubleCPTVertex cpt = createCPTWithMixedInputs(weapon, material);

        material.observe(GenericTensor.scalar(Material.WOOD));
        NetworkSamples samples = takeSamples(cpt, weapon, material);
        Map<Weapon, Double> weaponSampleProportion = calculateSampleProportions(samples, weapon);
        Assert.assertEquals(1.0, weaponSampleProportion.get(Weapon.CLUB), 0.001);
    }

    private CategoricalVertex<Weapon> getWeaponPrior() {
        Map<Weapon, DoubleVertex> weaponPrior = new HashMap<>();
        weaponPrior.put(Weapon.POISON, ConstantVertex.of(0.25));
        weaponPrior.put(Weapon.SPANNER, ConstantVertex.of(0.25));
        weaponPrior.put(Weapon.CLUB, ConstantVertex.of(0.25));
        weaponPrior.put(Weapon.KNIFE, ConstantVertex.of(0.25));
        return new CategoricalVertex<>(weaponPrior);
    }

    private CategoricalVertex<Material> getWeaponMaterialPrior() {
        Map<Material, DoubleVertex> weaponMaterialPrior = new HashMap<>();
        weaponMaterialPrior.put(Material.IRON, ConstantVertex.of(0.2));
        weaponMaterialPrior.put(Material.WOOD, ConstantVertex.of(0.2));
        weaponMaterialPrior.put(Material.STEAL, ConstantVertex.of(0.2));
        weaponMaterialPrior.put(Material.RUBBER, ConstantVertex.of(0.2));
        weaponMaterialPrior.put(Material.FUNNY_SMELL, ConstantVertex.of(0.2));
        return new CategoricalVertex<>(weaponMaterialPrior);
    }

    private DoubleCPTVertex createCPTWithBooleanInputs(CategoricalVertex<Weapon> weapon,
                                                       CategoricalVertex<Material> material) {

        BooleanVertex isPoison = new EqualsVertex<>(weapon, ConstantVertex.of(Weapon.POISON));
        BooleanVertex isSpanner = new EqualsVertex<>(weapon, ConstantVertex.of(Weapon.SPANNER));
        BooleanVertex isClub = new EqualsVertex<>(weapon, ConstantVertex.of(Weapon.CLUB));
        BooleanVertex isKnife = new EqualsVertex<>(weapon, ConstantVertex.of(Weapon.KNIFE));
        BooleanVertex isIron = new EqualsVertex<>(material, ConstantVertex.of(Material.IRON));
        BooleanVertex isWood = new EqualsVertex<>(material, ConstantVertex.of(Material.WOOD));
        BooleanVertex isSteal = new EqualsVertex<>(material, ConstantVertex.of(Material.STEAL));
        BooleanVertex isRubber = new EqualsVertex<>(material, ConstantVertex.of(Material.RUBBER));
        BooleanVertex isFunnySmell = new EqualsVertex<>(material, ConstantVertex.of(Material.FUNNY_SMELL));

        DoubleCPTVertex cpt = ConditionalProbabilityTable.of(isPoison, isSpanner, isClub, isKnife, isIron, isWood, isSteal, isRubber, isFunnySmell)
            .when(true, false, false, false, false, false, false, false, true).then(1.0)
            .when(false, true, false, false, true, false, false, false, false).then(0.5)
            .when(false, true, false, false, false, false, true, false, false).then(0.5)
            .when(false, false, true, false, true, false, false, false, false).then(0.25)
            .when(false, false, true, false, false, true, false, false, false).then(0.25)
            .when(false, false, true, false, false, false, true, false, false).then(0.25)
            .when(false, false, true, false, false, false, false, true, false).then(0.25)
            .when(false, false, false, true, true, false, false, false, false).then(0.3333333333333333)
            .when(false, false, false, true, false, false, true, false, false).then(0.3333333333333333)
            .when(false, false, false, true, false, false, false, true, false).then(0.3333333333333333)
            .orDefault(0.0);

        new BernoulliVertex(cpt).observe(true);

        return cpt;
    }

    private DoubleCPTVertex createCPTWithEnumInputs(CategoricalVertex<Weapon> weapon,
                                                    CategoricalVertex<Material> material) {

        DoubleCPTVertex cpt = ConditionalProbabilityTable.of(weapon, material)
            .when(Weapon.POISON, Material.FUNNY_SMELL).then(1.0)
            .when(Weapon.SPANNER, Material.IRON).then(0.5)
            .when(Weapon.SPANNER, Material.STEAL).then(0.5)
            .when(Weapon.CLUB, Material.IRON).then(0.25)
            .when(Weapon.CLUB, Material.WOOD).then(0.25)
            .when(Weapon.CLUB, Material.STEAL).then(0.25)
            .when(Weapon.CLUB, Material.RUBBER).then(0.25)
            .when(Weapon.KNIFE, Material.IRON).then(0.3333333333333333)
            .when(Weapon.KNIFE, Material.STEAL).then(0.3333333333333333)
            .when(Weapon.KNIFE, Material.RUBBER).then(0.3333333333333333)
            .orDefault(0.0);

        new BernoulliVertex(cpt).observe(true);
        return cpt;
    }

    private DoubleCPTVertex createCPTWithMixedInputs(CategoricalVertex<Weapon> weapon,
                                                     CategoricalVertex<Material> material) {

        BooleanVertex isPoison = new EqualsVertex<>(weapon, ConstantVertex.of(Weapon.POISON));
        BooleanVertex isSpanner = new EqualsVertex<>(weapon, ConstantVertex.of(Weapon.SPANNER));
        BooleanVertex isClub = new EqualsVertex<>(weapon, ConstantVertex.of(Weapon.CLUB));
        BooleanVertex isKnife = new EqualsVertex<>(weapon, ConstantVertex.of(Weapon.KNIFE));

        DoubleCPTVertex cpt = ConditionalProbabilityTable.of(isPoison, isSpanner, isClub, isKnife, material)
            .when(true, false, false, false, Material.FUNNY_SMELL).then(1.0)
            .when(false, true, false, false, Material.IRON).then(0.5)
            .when(false, true, false, false, Material.STEAL).then(0.5)
            .when(false, false, true, false, Material.IRON).then(0.25)
            .when(false, false, true, false, Material.WOOD).then(0.25)
            .when(false, false, true, false, Material.STEAL).then(0.25)
            .when(false, false, true, false, Material.RUBBER).then(0.25)
            .when(false, false, false, true, Material.IRON).then(0.3333333333333333)
            .when(false, false, false, true, Material.STEAL).then(0.3333333333333333)
            .when(false, false, false, true, Material.RUBBER).then(0.3333333333333333)
            .orDefault(0.0);

        new BernoulliVertex(cpt).observe(true);
        return cpt;
    }

    private NetworkSamples takeSamples(DoubleCPTVertex cpt, CategoricalVertex<Weapon> weapon,
                                       CategoricalVertex<Material> material) {

        BayesianNetwork bnet = new BayesianNetwork(cpt.getConnectedGraph());
        bnet.probeForNonZeroProbability(10000);
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(bnet);

        return Keanu.Sampling.MetropolisHastings.withDefaultConfig().generatePosteriorSamples(
            model,
            Arrays.asList(weapon, material)
        ).generate(11000).drop(1000).downSample(2);
    }

    private <T> Map<T, Double> calculateSampleProportions(NetworkSamples samples,
                                                          CategoricalVertex<T> inputCategorical) {

        List<T> inputSamples = samples.get(inputCategorical).asList().stream()
            .map(s -> s.scalar()).collect(Collectors.toList());

        return inputSamples.stream().collect(Collectors.groupingBy(Function.identity(),
            Collectors.collectingAndThen(Collectors.counting(), count -> ((double) count) / inputSamples.size())));
    }
}
