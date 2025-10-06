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
            using var network = new NeuralNetwork(n);
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
        using var network = new NeuralNetwork(_testing[0]);
        const double w = 0.5;

        network.SetWeight(0, 1, 1, w);
        Assert.That(network.GetWeight(0, 1, 1), Is.EqualTo(w));
    }

    [Test]
    public void PredictCountTest()
    {
        foreach (var n in _testing)
        {
            using var network = new NeuralNetwork(n);
            var input = new double[n[0]];
            var output = network.Predict(input);
            Assert.That(output.Length, Is.EqualTo(n[^1]));
        }
    }

    [Test]
    public void CostTest()
    {
        using var network = new NeuralNetwork(_testing[0]);
        
        Console.WriteLine(network.GetCost(new [] {1, 2.2}, new [] {1, 2.2}));
        
        var data = GenerateTestDataForSimpleLearnTest();
        data.Shuffle(5);
        var flattened = FlattenedData.FromArrays(data);
        
        using var network2 = new NeuralNetwork(_testing[3]);
        Console.WriteLine(network2.GetCost(flattened));
    }

    [Test]
    public void TestSetAllWeightsAndBiases()
    {
        foreach (var t in _testing)
        {
            using var n = new NeuralNetwork(t);
            const double w = 2.5;
            const double b = 3.4;
            n.SetAllWeightsAndBiases(w, b);

            for (int l = 0; l < n.Length; l++)
            {
                for (int o = 0; o < n.GetOutCount(l); o++)
                {
                    for (int i = 0; i < n.GetInCount(l); i++)
                    {
                        Assert.That(n.GetWeight(l, i, o), Is.EqualTo(w));
                    }
                    
                    Assert.That(n.GetBias(l, o), Is.EqualTo(b));
                }
            }
        }
    }

    [Test]
    public void LearnFromLittleFlattenedTest()
    {
        using var network = new NeuralNetwork(_testing[0]);

        var f = new FlattenedData(new[] { 1, 2.2, 3, 4 }, 2, new[] { 1.1, 2, 3, 4, 5, 6 }, 3);
        network.Learn(f, 1);
    }

    #region SimpleLearnTest

    private static readonly (int[], string)[] _configs =
    {
        (_testing[3], "NoMomentumSigmoid"),
    };

    [Test]
    public void SimpleLearnTest()
    {
        var data = GenerateTestDataForSimpleLearnTest();
        var flattened = FlattenedData.FromArrays(data);

        foreach (var config in _configs)
        {
            using var network = new NeuralNetwork(config.Item1);
            
            using var state = new LearningState(network);
            for (int i = 0; i < 1; i++)
            {
                network.Learn(flattened, 1000);
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
            
            Console.WriteLine($"{config.Item2} - {string.Join(",", config.Item1)} :: Accuracy : {accuracy}% - Cost : {cost}");
        }
    }

    [Test]
    public void SimpleCostTest()
    {
        var data = GenerateTestDataForSimpleLearnTest();
        var flattened = FlattenedData.FromArrays(data);
        
        using var network = new NeuralNetwork(_configs[0].Item1);
        network.SetAllWeightsAndBiases(1, 1);
        Console.WriteLine(network.GetCost(flattened));
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