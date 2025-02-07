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
    public void PredictTest()
    {
        foreach (var n in _testing)
        {
            using var network = new NeuralNetwork(n);
            var input = new double[n[0]];
            var output = network.Predict(input);
            Assert.That(output.Length, Is.EqualTo(n[^1]));
        }
    }
}