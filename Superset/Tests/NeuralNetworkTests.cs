using Model;

namespace Tests;

public class NeuralNetworkTests
{
    private static readonly int[][] _testing =
    {
        new[] { 2, 3, 2 },
        new[] { 1, 1, 1, 1, 1 },
        new[] { 2, 7, 4, 2 },
        new[] { 7, 4, 3 },
        new[] { 7, 5, 4, 3 }
    };

    [Test]
    public void LengthTest()
    {
        foreach (var n in _testing)
        {
            using var network = new NeuralNetwork(NeuralNetworkParameters.NoMomentumSigmoid, n);
            Assert.That(network.Length, Is.EqualTo(n.Length - 1));
            for (int i = 1; i < n.Length; i++)
            {
                Assert.That(network.GetInCount(i - 1), Is.EqualTo(n[i - 1]));
                Assert.That(network.GetOutCount(i - 1), Is.EqualTo(n[i]));
            }
        }
    }

    [Test]
    public void WeightsAndBiasesTest()
    {
        using var network = new NeuralNetwork(NeuralNetworkParameters.NoMomentumSigmoid, _testing[0]);
        const double w = 0.5;

        network.SetWeight(0, 1, 1, w);
        Assert.That(network.GetWeight(0, 1, 1), Is.EqualTo(w));
    }

    [Test]
    public void PredictCountTest()
    {
        foreach (var n in _testing)
        {
            using var network = new NeuralNetwork(NeuralNetworkParameters.NoMomentumSigmoid, n);
            var input = new double[n[0]];
            var output = network.Predict(input);
            Assert.That(output.Length, Is.EqualTo(n[^1]));
        }
    }

    [Test]
    public void CostTest()
    {
        using var network = new NeuralNetwork(NeuralNetworkParameters.NoMomentumSigmoid, _testing[0]);
        network.Randomize(0, 1);
        
        Console.WriteLine(network.GetCost(new [] {1, 2.2}, new [] {1, 2.2}));
        
        var data = GenerateTestDataForSimpleLearnTest();
        data.Shuffle(5);
        var flattened = FlattenedData.FromArrays(data);
        
        using var network2 = new NeuralNetwork(NeuralNetworkParameters.NoMomentumSigmoid, _testing[3]);
        network2.Randomize(0, 1);
        Console.WriteLine(network2.GetCost(flattened));
    }

    [Test]
    public void LearnFromLittleFlattenedTest()
    {
        using var network = new NeuralNetwork(NeuralNetworkParameters.NoMomentumSigmoid, _testing[0]);
        network.Randomize(0, 1);

        var f = new FlattenedData(new[] { 1, 2.2, 3, 4 }, 2, new[] { 1.1, 2, 3, 4, 5, 6 }, 3);
        network.Learn(f, 1, 1);
    }

    #region SimpleLearnTest

    private static readonly (NeuralNetworkParameters, int[], string)[] _configs =
    {
        (NeuralNetworkParameters.NoMomentumSigmoid, _testing[3], "NoMomentumSigmoid"),
        (NeuralNetworkParameters.MomentumSigmoid, _testing[3], "MomentumSigmoid"),
        (NeuralNetworkParameters.NoMomentumSigmoid, _testing[4], "NoMomentumSigmoid"),
        (NeuralNetworkParameters.MomentumSigmoid, _testing[4], "MomentumSigmoid")
    };

    [Test]
    public void SimpleLearnTest()
    {
        var data = GenerateTestDataForSimpleLearnTest();
        data.Shuffle(5);
        var flattened = FlattenedData.FromArrays(data);

        foreach (var config in _configs)
        {
            using var network = new NeuralNetwork(config.Item1, config.Item2);
            network.Randomize(0, 1);

            const int batchSize = 50;
            using var state = new LearningState(network);
            for (int i = 0; i < 5; i++)
            {
                network.Learn(flattened, batchSize, 5000, state);
            }

            var accuracy = 0.0;
            foreach (var (input, expected) in data)
            {
                var got = network.Predict(input);
                bool ok = true;
                for (int i = 0; i < got.Length; i++)
                {
                    if (Math.Abs(Math.Round(got[i]) - expected[i]) > 0.1) ok = false;
                }

                if (ok) accuracy++;
            }

            accuracy = accuracy / data.Length * 100;
            var cost = network.GetCost(flattened);
            
            Console.WriteLine($"{config.Item3} - {string.Join(",", config.Item2)} :: Accuracy : {accuracy}% - Cost : {cost}");
        }
    }

    private (double[], double[])[] GenerateTestDataForSimpleLearnTest()
    {
        List<(double[], double[])> result = new();
        GenerateTestDataForSimpleLearnTest(result, new double[7], 0);
        return result.ToArray();
    }

    private void GenerateTestDataForSimpleLearnTest(List<(double[], double[])> result, double[] current, int ind)
    {
        if (ind == 7)
        {
            var count = 0;
            foreach (var val in current) count += (int)val;

            var expected = new double[]
            {
                count >= 4 ? 1 : 0,
                count is 2 or 3 or 6 or 7 ? 1 : 0,
                count & 1
            };
                
            result.Add((current.Copy(), expected));
            return;
        }
        
        GenerateTestDataForSimpleLearnTest(result, current, ind + 1);
        current[ind] = 1;
        GenerateTestDataForSimpleLearnTest(result, current, ind + 1);
        current[ind] = 0;
    }

    #endregion
}