using Model;

namespace Tests;

public class NeuralNetworkTests
{
    private static readonly int[][] _testing =
    {
        new[] { 2, 3, 2 },
        new[] { 1, 1, 1, 1, 1 }
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
        network.Randomize(0, 1);
        
        Console.WriteLine(network.GetCost(new [] {1, 2.2}, new [] {1, 2.2}));
        
        var data = GenerateTestDataForSimpleLearnTest();
        data.Shuffle(5);
        var flattened = FlattenedData.FromArrays(data);
        
        using var network2 = new NeuralNetwork(7, 4, 3);
        network2.Randomize(0, 1);
        Console.WriteLine(network2.GetCost(flattened));
    }

    [Test]
    public void LearnFromLittleFlattenedTest()
    {
        using var network = new NeuralNetwork(_testing[0]);
        network.Randomize(0, 1);

        var f = new FlattenedData(new[] { 1, 2.2, 3, 4 }, 2, new[] { 1.1, 2, 3, 4, 5, 6 }, 3);
        network.Learn(f, 1, 1);
    }

    #region SimpleLearnTest

    [Test] //TODO fix
    public void SimpleLearnTest()
    {
        using var network = new NeuralNetwork(7, 4, 3);
        network.Randomize(0, 1);

        var data = GenerateTestDataForSimpleLearnTest();
        data.Shuffle(5);
        var flattened = FlattenedData.FromArrays(data);
        double[] predicted;

        const int tests = 50;
        for (int i = 0; i < tests; i++)
        {
            predicted = network.Predict(flattened.Inputs, 7);
            flattened.Inputs.Print(7);
            predicted.Print();
            Console.WriteLine();
            
            network.Learn(flattened, 50, 100);
        }
        
        predicted = network.Predict(flattened.Inputs, 7);
        flattened.Inputs.Print(7);
        predicted.Print();
        Console.WriteLine();
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